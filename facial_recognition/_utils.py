"""Utility Functions"""
# pylint: disable=E1101,E0401,C0413
from typing import List
import cv2
import dlib
import numpy as np

def draw_bounding_boxes(
        frame:np.ndarray,
        faces:List[dlib.rectangle]
    ):
    """Draws bounding boxes around detected faces in the given frame.

    Args:
        frame (np.ndarray): The input frame/image.
        faces (List[dlib.rectangle]): A list of dlib.rectangle objects representing the detected
                                      faces.

    Returns:
        None

    Example:
        frame = cv2.imread('image.jpg')
        faces = detector(frame)
        draw_bounding_boxes(frame, faces)
    """
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

def draw_landmarks(
        frame:np.ndarray,
        landmarks:List[dlib.point]
    ) -> None:
    """Draws landmarks on the given frame.

    Args:
        frame (np.ndarray): The frame on which the landmarks will be drawn.
        landmarks (List[dlib.point]): The list of landmarks to be drawn.

    Returns:
        None

    Note:
        This function iterates over each landmark and draws a circle at each landmark point on
        the frame.
        The circle is filled with green color and has a radius of 2 pixels.
    """
    for landmark in landmarks:
        for i in range(landmark.num_parts):
            x = landmark.part(i).x
            y = landmark.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def draw_name(
        frame:np.ndarray,
        face_rect:dlib.rectangle,
        name:str
    ) -> None:
    """Draws the name on the given frame at the bottom of the face rectangle.

    Args:
        frame (np.ndarray): The frame on which the name will be drawn.
        face_rect (dlib.rectangle): The rectangle representing the face.
        name (str): The name to be drawn.

    Returns:
        None: This function does not return anything.
    """
    x, y, _, h = face_rect.left(), face_rect.top(),\
                 face_rect.width(), face_rect.height()
    cv2.putText(frame, name, (x, y + h + 35), cv2.FONT_HERSHEY_SIMPLEX,\
                1.2, (0, 255, 0), 2)


def draw_fps(
        frame:np.ndarray,
        delta:float
    ) -> None:
    """Draws the frames per second (FPS) on the given frame.

    Args:
        frame (np.ndarray): The frame on which the FPS will be drawn.
        delta (float): The time difference between frames.

    Returns:
        None

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> delta = 0.033  # 30 FPS
        >>> draw_fps(frame, delta)
    """
    fps = 1 / delta
    cv2.putText(frame, f"FPS: {fps :02.1f}", (30, 30),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
