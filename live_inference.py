#!/usr/bin/env python3
"""
Simple webcam test script for verifying video feed (built-in or iPhone camera).
Press 'q' to quit.
"""

import cv2

def test_camera(camera_index=0, width=1280, height=720):
    print(f"🎥 Attempting to open camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Could not open camera index {camera_index}.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"✅ Camera {camera_index} opened successfully.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # Show the video feed
        cv2.imshow(f"Camera {camera_index} Feed", frame)

        # Exit cleanly on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera test ended.")


if __name__ == "__main__":
    # Try 0 (built-in) or 1 (iPhone / external)
    test_camera(camera_index=0)
