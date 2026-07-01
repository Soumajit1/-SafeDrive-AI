"""
camera_test.py — simple camera feed test
"""
import cv2
import os

def main():
    cap = None
    for i in [0, 1, 2]:
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if c.isOpened():
            cap = c
            print(f"[OK] Camera opened on index {i}")
            break

    if cap is None:
        print("[ERROR] No camera found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera test running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed")
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Smart Cabin AI — Camera Test  (Q to quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()