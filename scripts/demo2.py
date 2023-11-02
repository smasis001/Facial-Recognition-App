"""Demo Script #2"""
# pylint: disable=E1101,E0401,C0413
import os
import sys
import time
import cv2
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from facial_recognition import FaceDetection, draw_bounding_boxes, FaceAlignment,\
                               draw_landmarks, draw_fps,\
                               FaceRepresentation, FaceIdentification, draw_name

# Constants
CAMERA_ID = 0
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

def show_webcam_feed():
    """Shows the webcam feed and performs facial detection, landmark detection, face representation,
       and face identification.

    This function opens the webcam feed and continuously reads frames from it. It performs the
    following steps for each frame:
    - Detects faces in the frame using a face detection model.
    - Draws bounding boxes around the detected faces.
    - Aligns the faces using a landmark detection model.
    - Draws landmarks on the aligned faces.
    - Represents the faces using a face representation model.
    - Identifies the faces using a face identification model.
    - Draws the name of the best match for each face.

    The function also displays the frames with the detected faces, landmarks, and FPS (frames per
    second) on a window titled "Facial Identification". Press the Escape key to exit the
    function.

    Args:
        None

    Returns:
        None
    """
    cap = cv2.VideoCapture(CAMERA_ID)
    if WINDOW_WIDTH is not None and WINDOW_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    face_detector = FaceDetection()
    landmark_predictor = FaceAlignment()

    # Initializing two extra steps: Representation & Identifications
    face_descriptor = FaceRepresentation()
    face_identifier = FaceIdentification()

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

                # Executing two extra steps: Representation & Identification
                descriptors = face_descriptor.represent(frame_rgb, landmarks)
                best_distances, best_neighbors = face_identifier.identify(descriptors, k=1)
                for face, neighbors, distances in zip(faces, best_neighbors, best_distances):
                    best_match_name = f"{neighbors[0]} {distances[0]:.3f}"
                    draw_name(frame, face, best_match_name)

            draw_fps(frame, delta)
            cv2.imshow('Facial Identification', frame)

            key = cv2.waitKey(1)
            if key == 27:  # Escape key
                break

    cap.release()
    cv2.destroyAllWindows()

# Main execution block
if __name__ == '__main__':
    show_webcam_feed()
