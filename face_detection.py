"""
face_detection.py — standalone face detection window
Run directly:  python face_detection.py
"""

import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from detector import DriverDetector

def main():
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
    print("Smart Cabin AI — Face Detection  (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        result = det.process(frame)

        display = result.annotated_frame if result.annotated_frame is not None else frame

        status = f"Face: {'YES' if result.face_detected else 'NO'}  " \
                 f"EAR: {result.avg_ear:.3f}  " \
                 f"Blinks: {result.blink_count}  " \
                 f"Score: {result.drowsiness_score:.0f}  " \
                 f"Alert: {result.alert_level}"
        print(f"\r{status}", end="", flush=True)

        cv2.imshow("Smart Cabin AI — Face Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    cap.release()
    det.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()