"""Demo Script #1"""
# pylint: disable=E1101,E0401,C0413
import os
import sys
import time
import cv2
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from facial_recognition import FaceDetection, draw_bounding_boxes, FaceAlignment,\
                               draw_landmarks, draw_fps

# Constants
CAMERA_ID = 0
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

def show_webcam_feed():
    """Shows the webcam feed and detects faces and facial landmarks in real-time.

    This function opens the webcam feed and continuously reads frames from it. It performs the
    following steps for each frame:
    - Detects faces in the frame using a face detection model.
    - Draws bounding boxes around the detected faces.
    - Aligns the faces using a landmark detection model.
    - Draws landmarks on the aligned faces.

    The function also displays the frames with the detected faces, landmarks, and FPS (frames per
    second) on a window titled "Facial Detection". Press the Escape key to exit the
    function.

    Args:
        None

    Returns:
        None

    Example usage:
        show_webcam_feed()
    """
    cap = cv2.VideoCapture(CAMERA_ID)
    if WINDOW_WIDTH is not None and WINDOW_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    face_detector = FaceDetection()
    landmark_predictor = FaceAlignment()

    current_time = time.time()

    while True:
        ret, frame = cap.read()

        if ret:
            delta = time.time() - current_time
            delta = delta if delta != 0 else 0.000001
            current_time = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)

            if faces:
                draw_bounding_boxes(frame, faces)
                landmarks = landmark_predictor.align(frame_rgb, faces)
                draw_landmarks(frame, landmarks)

            draw_fps(frame, delta)
            cv2.imshow('Facial Detection', frame)

            key = cv2.waitKey(1)
            if key == 27:  # Escape key
                break

    cap.release()
    cv2.destroyAllWindows()

# Main execution block
if __name__ == '__main__':
    show_webcam_feed()
