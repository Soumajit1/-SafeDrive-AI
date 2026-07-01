"""
detector.py — Smart Cabin AI
Real-time driver monitoring using OpenCV Haar cascades (+ optional MediaPipe).
Returns structured DetectionResult objects — no dummy data.

v2.1 fixes:
  - Eyes-closed and head-turned-away are now tracked as INDEPENDENT timers.
    Previously the drowsiness score was driven only by `eye_closed_frames`,
    which depends on the eye cascade actually finding eye shapes — but a
    turned head also makes the eye cascade find nothing, so the two cases
    got tangled together and head-turns alone often failed to raise the
    score.
  - The Haar fallback no longer treats "zero eyes found" as a fixed
    "almost closed" guess (0.18). It now distinguishes:
      * face is roughly frontal + no eyes found  -> eyes are CLOSED
      * face is turned away (large yaw)           -> don't trust eye state,
                                                       rely on head-turn timer
  - `distracted` (sustained head turn) now contributes its own score on
    top of the eye-closed score, so looking away long enough raises the
    gauge even with eyes wide open.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


# ─── MediaPipe Face Mesh indices for eyes ───────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
NOSE_TIP_IDX  = 1
CHIN_IDX      = 199


# ─── Data container ─────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    face_detected:    bool  = False
    left_ear:         float = 0.0
    right_ear:        float = 0.0
    avg_ear:          float = 0.0
    blink_detected:   bool  = False
    blink_count:      int   = 0
    eye_closed_frames:int   = 0
    head_turn_frames: int   = 0
    drowsiness_score: float = 0.0
    alert_level:      str   = "OK"       # "OK" | "FATIGUE" | "DROWSY"
    head_yaw:         float = 0.0        # degrees, + = right
    head_pitch:       float = 0.0        # degrees, + = down
    distracted:       bool  = False
    eyes_closed:      bool  = False
    fps:              float = 0.0
    detection_mode:   str   = "haar"     # "mediapipe" | "haar" — which path produced this frame
    annotated_frame:  Optional[np.ndarray] = field(default=None, repr=False)


# ─── Detector class ─────────────────────────────────────────────────────────
class DriverDetector:
    """
    Uses OpenCV Haar cascades for face + eye detection, with an optional
    MediaPipe FaceLandmarker path for precise EAR/head-pose when available.
    """

    EAR_THRESHOLD       = 0.22   # below this -> eye considered closed
    BLINK_CONSEC_MIN    = 2      # min frames closed = a blink
    BLINK_CONSEC_MAX    = 6      # above this -> sustained closure, not a blink
    DROWSY_EYE_CONSEC    = 20    # eye-closed frames -> eye score saturates (100)
    DROWSY_HEAD_CONSEC   = 25    # head-turned frames -> head score saturates (100)
    HEAD_DISTRACT_DEG    = 22    # yaw degrees off-centre -> counted as "turned away"
    HEAD_FRONTAL_DEG     = 15    # below this yaw, trust the eye cascade fully

    # Final score blends both signals: max() so either alone can trigger DROWSY,
    # with a small bonus if both are happening simultaneously.
    EYE_SCORE_WEIGHT  = 1.0
    HEAD_SCORE_WEIGHT = 1.0

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.eye_tree_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

        self._mp_landmarker = None
        self._try_init_mediapipe()

        # State
        self._blink_count        = 0
        self._eye_closed_frames  = 0
        self._head_turn_frames   = 0
        self._fps_ts             = time.time()
        self._fps_frame_count    = 0
        self._fps                = 0.0
        # Smoothing for EAR to reduce single-frame cascade flicker
        self._ear_smooth         = 0.30

        # Haar-fallback reliability tracking (see _haar_eye_analysis):
        # a single "no eyes found" frame is NOT enough evidence of closed
        # eyes on its own — glasses/lighting/JPEG compression regularly
        # defeat the eye cascade even with eyes wide open.
        self._frontal_miss_streak = 0
        self._eye_hit_history     = deque(maxlen=30)

    # ── MediaPipe optional setup ─────────────────────────────────────────
    def _try_init_mediapipe(self):
        try:
            import mediapipe as mp
            import os
            model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if not os.path.exists(model_path):
                print("[Detector] face_landmarker.task not found — using Haar cascade only.")
                return
            BaseOptions           = mp.tasks.BaseOptions
            FaceLandmarker        = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_landmarker = FaceLandmarker.create_from_options(opts)
            print("[Detector] MediaPipe FaceLandmarker loaded ✓ (high-precision mode)")
        except Exception as e:
            print(f"[Detector] MediaPipe not available ({e}) — using Haar cascade EAR fallback.")

    # ── EAR from 6 landmark points ───────────────────────────────────────
    @staticmethod
    def _ear(pts: np.ndarray) -> float:
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.3

    # ── Main process frame ───────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> DetectionResult:
        result = DetectionResult()
        self._fps_frame_count += 1
        now = time.time()
        if now - self._fps_ts >= 1.0:
            self._fps = self._fps_frame_count / (now - self._fps_ts)
            self._fps_frame_count = 0
            self._fps_ts = now
        result.fps = round(self._fps, 1)

        if frame is None or frame.size == 0:
            return result

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # No face at all -> can't say anything about eyes, but DO NOT
            # silently reset timers to zero; a driver who turns far enough
            # that the face cascade also loses them is still a risk.
            # We keep counting head-turn frames (treat "face lost" like
            # "looking away"), but we don't count eye-closed frames since
            # we have no eye information at all.
            self._head_turn_frames = min(self._head_turn_frames + 1,
                                          self.DROWSY_HEAD_CONSEC * 2)
            result.head_turn_frames = self._head_turn_frames
            result.distracted = True
            result.detection_mode = "mediapipe" if self._mp_landmarker else "haar"
            self._finalize_score(result, eyes_trustworthy=False)
            result.annotated_frame = self._overlay(display, result)
            return result

        result.face_detected = True

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), (0, 220, 100), 2)
        cv2.putText(display, "Driver", (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1)

        # ── Head pose first (needed to judge whether eye cascade is trustworthy)
        self._estimate_head_pose_haar(gray, result, fx, fy, fw, fh)

        # ── Try MediaPipe for precise EAR + head pose (overrides Haar pose) ──
        mp_ok = self._mediapipe_analysis(frame, display, result, fx, fy, fw, fh)

        eyes_trustworthy = True
        if not mp_ok:
            eyes_trustworthy = self._haar_eye_analysis(gray, display, result, fx, fy, fw, fh)
        result.detection_mode = "mediapipe" if mp_ok else "haar"

        # ── Update independent timers ─────────────────────────────────
        self._update_eye_timer(result, eyes_trustworthy)
        self._update_head_timer(result)

        # ── Combine into final score / alert level ──────────────────
        self._finalize_score(result, eyes_trustworthy=eyes_trustworthy)

        result.annotated_frame = self._overlay(display, result)
        return result

    # ── MediaPipe path (precise EAR + head pose) ──────────────────────────
    def _mediapipe_analysis(self, frame, display, result, fx, fy, fw, fh) -> bool:
        if self._mp_landmarker is None:
            return False
        try:
            import mediapipe as mp
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_res = self._mp_landmarker.detect(mp_img)
            if not mp_res.face_landmarks:
                return False

            h, w = frame.shape[:2]
            lms  = mp_res.face_landmarks[0]

            def pt(idx):
                l = lms[idx]
                return np.array([l.x * w, l.y * h])

            l_pts = np.array([pt(i) for i in LEFT_EYE_IDX])
            r_pts = np.array([pt(i) for i in RIGHT_EYE_IDX])

            left_ear  = self._ear(l_pts)
            right_ear = self._ear(r_pts)
            avg_ear   = (left_ear + right_ear) / 2

            self._ear_smooth = 0.5 * self._ear_smooth + 0.5 * avg_ear

            result.left_ear  = round(left_ear, 3)
            result.right_ear = round(right_ear, 3)
            result.avg_ear   = round(self._ear_smooth, 3)

            for p in l_pts:
                cv2.circle(display, tuple(p.astype(int)), 2, (255, 180, 0), -1)
            for p in r_pts:
                cv2.circle(display, tuple(p.astype(int)), 2, (255, 180, 0), -1)

            nose    = pt(NOSE_TIP_IDX)
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            norm_x  = (nose[0] - face_cx) / (fw / 2)
            norm_y  = (nose[1] - face_cy) / (fh / 2)
            result.head_yaw   = round(norm_x * 45, 1)
            result.head_pitch = round(norm_y * 30, 1)
            result.distracted = abs(result.head_yaw) > self.HEAD_DISTRACT_DEG

            return True
        except Exception:
            return False

    # ── Haar head-pose estimate (always runs as a baseline) ───────────────
    def _estimate_head_pose_haar(self, gray, result, fx, fy, fw, fh):
        h_frame, w_frame = gray.shape[:2]
        face_cx = fx + fw / 2
        face_cy = fy + fh / 2
        result.head_yaw   = round(((face_cx / w_frame) - 0.5) * 70, 1)
        result.head_pitch = round(((face_cy / h_frame) - 0.42) * 45, 1)
        result.distracted = abs(result.head_yaw) > self.HEAD_DISTRACT_DEG

    # ── Haar eye-open/closed analysis ──────────────────────────────────────
    def _haar_eye_analysis(self, gray, display, result, fx, fy, fw, fh) -> bool:
        """
        Returns eyes_trustworthy: False when the head is turned far enough
        that "no eyes detected" should NOT be interpreted as "eyes closed"
        (because the cascade can't see eye shapes from a profile angle
        anyway). When True, the eye-closed reading is meaningful and should
        drive the eye-closed timer.
        """
        face_is_frontal = abs(result.head_yaw) <= self.HEAD_FRONTAL_DEG

        roi_y1, roi_y2 = fy, fy + int(fh * 0.62)
        roi_gray = gray[roi_y1:roi_y2, fx:fx+fw]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8,
            minSize=(18, 18)
        )
        if len(eyes) == 0:
            eyes = self.eye_tree_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(15, 15)
            )

        eyes = sorted(eyes, key=lambda r: r[2]*r[3], reverse=True)[:2]

        open_eye_count = 0
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            is_open = self._eye_roi_is_open(eye_roi)
            if is_open:
                open_eye_count += 1
            color = (0, 200, 255) if is_open else (0, 0, 255)
            cv2.rectangle(display,
                          (fx+ex, roi_y1+ey),
                          (fx+ex+ew, roi_y1+ey+eh), color, 1)

        n_found = len(eyes)

        # Track how often the cascade finds ANY eyes at all on frontal
        # frames. If it's rarely finding eyes for this driver (glasses
        # glare, backlight, JPEG compression softness), that's a sign the
        # cascade just doesn't work well for them — not that their eyes
        # are actually closed every frame.
        if face_is_frontal:
            self._eye_hit_history.append(1 if n_found > 0 else 0)
        hit_rate = (
            sum(self._eye_hit_history) / len(self._eye_hit_history)
            if self._eye_hit_history else 1.0
        )

        if n_found == 0:
            if face_is_frontal:
                self._frontal_miss_streak += 1
                # Only trust "no eyes found" as "eyes closed" once we've
                # missed several consecutive frames (a single-frame miss is
                # noise, not a blink) AND the cascade has a reasonable
                # recent hit-rate overall (i.e. it normally CAN see this
                # driver's eyes — so a real closure, not glasses/lighting
                # defeating the cascade every single frame).
                if self._frontal_miss_streak >= 3 and hit_rate >= 0.35:
                    synth_ear = 0.14
                    eyes_trustworthy = True
                else:
                    synth_ear = max(self._ear_smooth, self.EAR_THRESHOLD + 0.02)
                    eyes_trustworthy = False
            else:
                # Head is turned -> cascade naturally fails regardless of
                # eye state. Don't claim eyes are closed; just hold the
                # last known EAR so the eye-closed timer doesn't move.
                self._frontal_miss_streak = 0
                synth_ear = max(self._ear_smooth, self.EAR_THRESHOLD + 0.02)
                eyes_trustworthy = False
        else:
            self._frontal_miss_streak = 0
            ear_map = {0: 0.14, 1: 0.20, 2: 0.30}
            synth_ear = ear_map.get(open_eye_count, 0.14)
            eyes_trustworthy = True

        self._ear_smooth = 0.4 * self._ear_smooth + 0.6 * synth_ear

        result.left_ear  = round(self._ear_smooth, 3)
        result.right_ear = round(self._ear_smooth, 3)
        result.avg_ear   = round(self._ear_smooth, 3)

        return eyes_trustworthy

    @staticmethod
    def _eye_roi_is_open(eye_roi_gray: np.ndarray) -> bool:
        """Heuristic: an open eye shows a dark pupil/iris blob against the
        sclera; a closed eye is mostly uniform mid-tone eyelid skin."""
        if eye_roi_gray.size == 0:
            return True
        roi = cv2.equalizeHist(eye_roi_gray)
        _, thresh = cv2.threshold(roi, 55, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(thresh == 255) / thresh.size
        return 0.04 < dark_ratio < 0.40

    # ── Independent timers ─────────────────────────────────────────────────
    def _update_eye_timer(self, result: DetectionResult, eyes_trustworthy: bool):
        eye_closed = eyes_trustworthy and (result.avg_ear < self.EAR_THRESHOLD)
        result.eyes_closed = eye_closed

        if eye_closed:
            self._eye_closed_frames += 1
        else:
            if self.BLINK_CONSEC_MIN <= self._eye_closed_frames <= self.BLINK_CONSEC_MAX:
                self._blink_count += 1
                result.blink_detected = True
            if eyes_trustworthy:
                self._eye_closed_frames = 0

        result.blink_count       = self._blink_count
        result.eye_closed_frames = self._eye_closed_frames

    def _update_head_timer(self, result: DetectionResult):
        if result.distracted:
            self._head_turn_frames += 1
        else:
            self._head_turn_frames = max(0, self._head_turn_frames - 2)
        result.head_turn_frames = self._head_turn_frames

    # ── Combine eye + head signals into final score ────────────────────────
    def _finalize_score(self, result: DetectionResult, eyes_trustworthy: bool):
        eye_score  = min(100.0, (self._eye_closed_frames / self.DROWSY_EYE_CONSEC) * 100) * self.EYE_SCORE_WEIGHT
        head_score = min(100.0, (self._head_turn_frames / self.DROWSY_HEAD_CONSEC) * 100) * self.HEAD_SCORE_WEIGHT

        combined = max(eye_score, head_score)
        if eye_score > 15 and head_score > 15:
            combined = min(100.0, combined + min(eye_score, head_score) * 0.25)

        result.drowsiness_score = round(combined, 1)

        if result.drowsiness_score >= 70:
            result.alert_level = "DROWSY"
        elif result.drowsiness_score >= 35:
            result.alert_level = "FATIGUE"
        else:
            result.alert_level = "OK"

    # ── HUD overlay ───────────────────────────────────────────────────────
    def _overlay(self, frame: np.ndarray, r: DetectionResult) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status_color = {
            "OK":      (50, 220, 80),
            "FATIGUE": (30, 170, 255),
            "DROWSY":  (30, 30, 255),
        }.get(r.alert_level, (200, 200, 200))

        status_text = {
            "OK":      "✓  ALERT",
            "FATIGUE": "⚠  FATIGUE",
            "DROWSY":  "⛔  DROWSY",
        }.get(r.alert_level, r.alert_level)

        cv2.putText(frame, f"Smart Cabin AI  |  {status_text}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"FPS:{r.fps:.0f}  EAR:{r.avg_ear:.2f}  "
                           f"Blinks:{r.blink_count}  Yaw:{r.head_yaw:.0f}°  "
                           f"EyeCl:{r.eye_closed_frames}  HeadT:{r.head_turn_frames}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        return frame

    def release(self):
        if self._mp_landmarker:
            try:
                self._mp_landmarker.close()
            except Exception:
                pass