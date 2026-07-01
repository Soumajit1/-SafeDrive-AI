"""
server.py — Smart Cabin AI backend.

Architecture:
  Browser (getUserMedia) -> captures webcam frames -> sends JPEG over WebSocket
  Flask + flask-sock      -> receives frame -> runs DriverDetector (detector.py)
                           -> sends back JSON telemetry + annotated JPEG frame
  Browser                 -> renders annotated frame + live metrics on dashboard

Run:
  python server.py
Then open:
  http://localhost:5000
"""

import sys
import os
import time
import json
import base64

import cv2
import numpy as np
from flask import Flask, render_template, jsonify
from flask_sock import Sock

sys.path.insert(0, os.path.dirname(__file__))
from detector import DriverDetector

app = Flask(__name__)
sock = Sock(app)

# One detector instance per server process (single-driver kiosk use case).
# For multi-user concurrent use, this would need per-session detector instances.
detector = DriverDetector()


# ───────────────────────── Page routes ─────────────────────────────────────
@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/permission")
def permission():
    return render_template("permission.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "smart-cabin-ai"})


# ───────────────────────── WebSocket: live video pipeline ──────────────────
@sock.route("/ws/stream")
def stream(ws):
    """
    Protocol:
      Client -> Server: { "frame": "<base64 jpeg>" }
      Server -> Client : {
          "frame": "<base64 jpeg annotated>",
          "face_detected": bool,
          "left_ear": float, "right_ear": float, "avg_ear": float,
          "blink_count": int, "eye_closed_frames": int,
          "drowsiness_score": float, "alert_level": "OK"|"FATIGUE"|"DROWSY",
          "head_yaw": float, "head_pitch": float, "distracted": bool,
          "fps": float
      }
      Client -> Server: { "cmd": "reset" }   -> resets blink/drowsiness counters
    """
    print("[WS] client connected")
    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("cmd") == "reset":
                detector._blink_count = 0
                detector._eye_closed_frames = 0
                ws.send(json.dumps({"reset": True}))
                continue

            b64 = msg.get("frame")
            if not b64:
                continue

            # Decode incoming JPEG
            try:
                img_bytes = base64.b64decode(b64.split(",")[-1])
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[WS] decode error: {e}")
                ws.send(json.dumps({"error": f"frame decode failed: {e}"}))
                continue

            if frame is None:
                ws.send(json.dumps({"error": "frame decode returned empty image"}))
                continue

            try:
                result = detector.process(frame)
            except Exception as e:
                print(f"[WS] detector error: {e}")
                ws.send(json.dumps({"error": f"detector failed: {e}"}))
                continue






            # Encode annotated frame back to JPEG/base64
                       # Encode annotated frame back to JPEG/base64
            out_frame = result.annotated_frame if result.annotated_frame is not None else frame
            ok, buf = cv2.imencode(
                ".jpg",
                out_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            out_b64 = base64.b64encode(buf).decode("utf-8") if ok else ""

            payload = {
                "frame": f"data:image/jpeg;base64,{out_b64}",
                "face_detected": bool(result.face_detected),
                "left_ear": float(result.left_ear),
                "right_ear": float(result.right_ear),
                "avg_ear": float(result.avg_ear),
                "blink_count": int(result.blink_count),
                "eye_closed_frames": int(result.eye_closed_frames),
                "drowsiness_score": float(result.drowsiness_score),
                "alert_level": str(result.alert_level),
                "head_yaw": float(result.head_yaw),
                "head_pitch": float(result.head_pitch),
                "distracted": bool(result.distracted),
                "fps": float(result.fps),
                "blink_detected": bool(result.blink_detected),
                "ts": time.strftime("%H:%M:%S"),
            }

            ws.send(json.dumps(payload))





    except Exception as e:
        print(f"[WS] connection closed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Cabin AI — server starting")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    # IMPORTANT: use_reloader must stay False. Flask's debug reloader
    # spawns a child process to actually serve requests, and flask-sock's
    # WebSocket connections get bound inconsistently across that
    # parent/child boundary — the browser's WebSocket reports "open"
    # (the TCP handshake succeeds) but no messages ever flow afterward,
    # which looks exactly like "camera ready, connecting to detector…"
    # hanging forever with no error. debug=True (for tracebacks in the
    # browser) is safe to keep; use_reloader=False is the part that matters.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
