"""Detect Faces Functionality"""
# pylint: disable=E1101,E0401,C0413
from typing import List
import dlib
import numpy as np
import mediapipe as mp

class FaceDetection:
    """Class for Face Detection"""
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.detector = mp_face_detection.\
                                FaceDetection(min_detection_confidence=0.5)

    def detect(
            self,
            img_rgb:np.ndarray
        ) -> List[dlib.rectangle]:
        """Detects faces in an RGB image using mediapipe library but storing in dlib format.
        
        Args:
            img_rgb (np.ndarray): The input RGB image.
        
        Returns:
            List[dlib.rectangle]: A list of dlib rectangles representing the detected faces.
        """
        faces = self.detector.process(img_rgb)
        img_h, img_w, _ = img_rgb.shape
        dlib_faces = []
        if faces.detections:
            for face in faces.detections:
                bbox = face.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
                w, h = int(bbox.width * img_w), int(bbox.height * img_h)
                dlib_faces.append(dlib.rectangle(x1, y1, x1+w, y1+h))
        return dlib_faces
