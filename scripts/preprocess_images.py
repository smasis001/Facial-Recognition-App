"""Pre-process Image Script"""
# pylint: disable=E1101,E0401,C0413
from typing import List, Optional
import os
import sys
import shutil
from pkg_resources import resource_filename
import cv2
import dlib
import numpy as np
from tqdm import tqdm
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from facial_recognition import FaceDetection, FaceAlignment, FaceRepresentation

# Constants
RAW_IMAGES_PATH = resource_filename(__name__,\
                               '../data/raw')
PROCESSED_IMAGES_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/processed')

def crop_faces(
        img:np.ndarray,
        faces:List[dlib.rectangle],
        padding:Optional[float] = 0.35
    ) -> List[np.ndarray]:
    """Crop faces from an image.

    Args:
        img (numpy.ndarray): The input image.
        faces (List[dlib.rectangle]): A list of face objects detected in the image.
        padding (Optional[float]): The padding factor to apply when cropping the faces. Defaults
                                   to 0.35.

    Returns:
        list: A list of cropped face images.

    Example:
        img = cv2.imread('image.jpg')
        faces = detect_faces(img)
        cropped_faces = crop_faces(img, faces)
    """
    #mult = 1 + padding
    mh, mw = img.shape[0], img.shape[1]

    cropped_faces = []

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        wh = max(w, h)
        ow = max(0, int(x - (wh * (padding/2)))) + int(wh * (1 + padding)) - mw
        oh = max(0, int(y - (wh * (padding/2)))) + int(wh * (1 + padding)) - mh
        if ow > 0 or oh > 0:
            pad = (max(ow, oh) / wh) * 2
        else:
            pad = padding
        shift = wh * (pad/2)
        whp = int(wh * (1 + pad))
        x1, y1 = max(0, int(-shift + x)), max(0, int(-shift + y))
        x2, y2 = x1 + whp, y1 + whp

        cropped_face = img[y1:y2, x1:x2]
        cropped_faces.append(cropped_face)

    return cropped_faces

def process_images(
        input_dir:str,
        output_dir:str
    ) -> None:
    """Process images in the input directory and save the cropped faces in the output directory.

    Args:
        input_dir (str): Path to the input directory containing the images.
        output_dir (str): Path to the output directory where the cropped faces will be saved.

    Returns:
        None
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    face_detector = FaceDetection()
    landmark_predictor = FaceAlignment()
    face_descriptor = FaceRepresentation()
    embeddings = []
    paths = []

    num_dirs = len(os.listdir(input_dir)) + 1
    for root, _, files in tqdm(os.walk(input_dir), total=num_dirs):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)
                img = cv2.imread(input_image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = face_detector.detect(img_rgb)
                else:
                    faces = None
                if isinstance(faces, list) and len(faces) == 1:
                    landmarks = landmark_predictor.align(img_rgb, faces)
                    if isinstance(landmarks, list) and len(landmarks)==1:
                        descriptors = face_descriptor.represent(img_rgb, landmarks)
                        if isinstance(descriptors, list) and len(descriptors) == 1:
                            relative_path = os.path.relpath(root, input_dir)
                            output_subdir = os.path.join(output_dir, relative_path)
                            os.makedirs(output_subdir, exist_ok=True)

                            cropped_faces = crop_faces(img, faces)

                            for idx, cropped_face in enumerate(cropped_faces):
                                output_image_path = os.path.join(output_subdir,\
                                                            f"{os.path.splitext(file)[0]}_face{idx}"
                                                            f"{os.path.splitext(file)[1]}")
                                cv2.imwrite(output_image_path, cropped_face)
                                print(f"Saved: {output_image_path}")
                                embeddings.append(descriptors[0])
                                paths.append(output_image_path.replace(output_dir, ""))
                        else:
                            print(f"Delete: {input_image_path}")
                            os.remove(input_image_path)
                    else:
                        print(f"Delete: {input_image_path}")
                        os.remove(input_image_path)
                else:
                    print(f"Delete: {input_image_path}")
                    os.remove(input_image_path)
    embeddings = np.array(embeddings, dtype='f')
    face_descriptor.add_to_vectordb(embeddings, ids=paths)

# Main execution block
if __name__ == '__main__':
    process_images(RAW_IMAGES_PATH, PROCESSED_IMAGES_PATH)
