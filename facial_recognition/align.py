"""Align Faces Functionality"""
# pylint: disable=E1101,E0401,C0413
from typing import List
import dlib
import numpy as np
import mediapipe as mp

MP_LANDMARK_SUBSET = 'large'
SUBSET_68_IDXS = [127, 234, 93, 215, 172, 136, 150, 176, 152, 400, 379,\
                365, 367, 433, 366, 447, 372, 70, 63, 105, 66, 107, 336,\
                296, 334, 293, 276, 168, 197, 195, 4, 240, 97, 2, 326,\
                290, 33, 160, 158, 133, 153, 144, 362, 385, 386, 249, 373,\
                380, 61, 39, 37, 11, 267, 269, 291, 321, 314, 17, 85, 181,\
                78, 82, 13, 402, 308, 402, 14, 87]
SUBSET_5_IDXS = [249, 362, 33, 155, 2]

class FaceAlignment:
    """Class for Face Alignment (aka Landmark Predition)"""
    def __init__(
            self,
            landmark_subset = MP_LANDMARK_SUBSET
        ) -> None:
        self.landmark_subset = landmark_subset
        mp_face_mesh = mp.solutions.face_mesh
        self.predictor = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,\
                                            refine_landmarks=True, min_detection_confidence=0.5)

    def align(
            self,
            img_rgb:np.ndarray,
            faces:List[dlib.rectangle]
        ) -> List[dlib.full_object_detection]:
        """Aligns the landmarks of detected faces in an image.

        Args:
            img_rgb (np.ndarray): The input RGB image.
            faces (List[dlib.rectangle]): The list of detected face rectangles.

        Returns:
            List[dlib.full_object_detection]: The list of aligned face landmarks.

        Raises:
            None

        Note:
            - The function uses the `landmark_subset` attribute of the class to determine the
              subset of landmarks to use.
            - If `landmark_subset` is set to 'large', the function uses the 68-point landmark
              subset.
            - If `landmark_subset` is set to 'small', the function uses the 5-point landmark
              subset.
            - If `landmark_subset` is set to any other value, the function uses the full 468-point
              landmark subset.

        Example:
            align(img_rgb, faces)
        """

        if self.landmark_subset == 'large':
            subset_idxs = SUBSET_68_IDXS
        elif self.landmark_subset == 'small':
            subset_idxs = SUBSET_5_IDXS
        else:
            subset_idxs = [*range(0, 468)]

        raw_landmarks = self.predictor.process(img_rgb)
        h, w, _ = img_rgb.shape
        dlib_landmarks = []
        if raw_landmarks.multi_face_landmarks:
            for f, face_landmarks in enumerate(raw_landmarks.multi_face_landmarks):
                dlib_parts = []
                for p in subset_idxs:
                    pt = face_landmarks.landmark[p]
                    dlib_parts.append(dlib.point(round(pt.x*w), round(pt.y*h)))
                dlib_fod = dlib.full_object_detection(rect=faces[f], parts=dlib_parts)
                dlib_landmarks.append(dlib_fod)

        return dlib_landmarks
