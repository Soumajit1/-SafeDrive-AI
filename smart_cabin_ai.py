"""
smart_cabin_ai.py — full MediaPipe FaceLandmarker version.
Requires face_landmarker.task in the same directory.

Download the model:
  curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
       -o face_landmarker.task

Run:  python smart_cabin_ai.py
"""

import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from detector import DriverDetector

def main():
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("[WARN] face_landmarker.task not found.")
        print("       Download it with:")
        print("       curl -L https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
              "-o face_landmarker.task")
        print("       Falling back to Haar cascade detector...")

    det = DriverDetector()
    cap = None
    for i in [0, 1, 2]:
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if c.isOpened():
            cap = c
            break

    if cap is None:
        print("[ERROR] No camera found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Smart Cabin AI — Full Detection  (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        result = det.process(frame)
        display = result.annotated_frame if result.annotated_frame is not None else frame

        cv2.imshow("Smart Cabin AI", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    det.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()