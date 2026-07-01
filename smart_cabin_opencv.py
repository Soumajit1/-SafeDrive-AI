"""
smart_cabin_opencv.py — real-time blink detection using OpenCV only.
Run directly:  python smart_cabin_opencv.py
"""

import cv2
import os
import sys
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
    print("Smart Cabin AI — Blink Detection  (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        result = det.process(frame)

        display = result.annotated_frame if result.annotated_frame is not None else frame

        # Extra drowsiness overlay
        if result.alert_level == "DROWSY":
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.putText(display, "!! DROWSINESS ALERT !!", (60, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Smart Cabin AI — Blink Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    det.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()