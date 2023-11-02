"""Represent Faces (with Descriptors) and store in Vector DB Functionality"""
# pylint: disable=E1101,E0401,C0413
from typing import Literal, List, Union
import os
from pkg_resources import resource_filename
import dlib
import faiss
import joblib
import numpy as np

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/vectordb/faces_l2.faiss')
VECTOR_KEYS_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/vectordb/faces_l2.pkl')
VECTOR_METRIC = 'euclidean'
VECTOR_DIMENSIONS = 128
DLIB_FACE_RECOGNITION_MDL_PATH = resource_filename(__name__,\
                                        '../models/dlib_face_recognition_resnet_model_v1.dat')
class FaceRepresentation:
    """Class for Face Representation (aka Face Descriptor)"""
    def __init__(
            self,
            metric:Literal['euclidean', 'cosine'] = VECTOR_METRIC,
            dimensions:int = VECTOR_DIMENSIONS
        ) -> None:
        self.recognizer = dlib.face_recognition_model_v1(DLIB_FACE_RECOGNITION_MDL_PATH)
        self.metric = metric
        self.dimensions = dimensions
        self.vectordb = None
        self.vectorkeys = []
        self.num_records = 0
        if os.path.exists(VECTOR_DB_PATH):
            self.vectordb:faiss.IndexFlat = faiss.read_index(VECTOR_DB_PATH)
            self.metric = 'euclidean' if self.vectordb.metric_type == 1 else 'cosine'
            self.num_records = self.vectordb.ntotal
            if os.path.exists(VECTOR_KEYS_PATH):
                self.vectorkeys = joblib.load(VECTOR_KEYS_PATH)
        else:
            if self.metric == 'euclidean':
                self.vectordb:faiss.IndexFlat = faiss.IndexFlatL2(dimensions)
            elif self.metric == 'cosine':
                self.vectordb:faiss.IndexFlat = faiss.IndexFlatIP(dimensions)

    def represent(
            self,
            img_rgb:np.ndarray,
            landmarks:List[dlib.full_object_detection]
        ) -> List[np.ndarray]:
        """Represent the given image with facial descriptors given their facial landmarks.

        Args:
            img_rgb (np.ndarray): The RGB image to represent.
            landmarks (List[dlib.full_object_detection]): The list of facial landmarks.

        Returns:
            List[np.ndarray]: The list of descriptors representing the image with facial landmarks.
        """
        descriptors = [np.array(self.recognizer.compute_face_descriptor(img_rgb, landmark,\
                                                                            num_jitters=1))\
                       for landmark in landmarks]
        return descriptors

    def add_to_vectordb(
            self,
            embeddings:np.ndarray,
            ids:List[Union[str,int]] = None
        ) -> None:
        """Adds embeddings to the VectorDB.

        Args:
            embeddings (np.ndarray): The embeddings to be added to the VectorDB.
            ids (List[Union[str,int]], optional): The IDs associated with the embeddings.
                                                  Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If the length of `ids` is not equal to the length of `embeddings`.

        Note:
            If `ids` is not provided, it will generate IDs automatically.

        Example:
            add_to_vectordb(embeddings, ids)
        """
        if (self.num_records==0) and (self.metric == 'cosine'):
            faiss.normalize_L2(embeddings)
        emb_len = embeddings.shape[0]
        new_total = self.num_records + emb_len
        self.vectordb.add(embeddings)
        if ids is None:
            ids = self.vectorkeys([*range(self.num_records,new_total)])
        else:
            assert emb_len == len(ids)
        self.vectorkeys.extend(ids)
        joblib.dump(self.vectorkeys, VECTOR_KEYS_PATH)
        faiss.write_index(self.vectordb, VECTOR_DB_PATH)
        self.num_records = new_total

    def convert_landmarks(
            self,
            bb:List[int],
            face_landmarks:List[List[int]]
        ) -> List[dlib.full_object_detection]:
        """Converts bounding box and face landmarks from lists to dlib.full_object_detection.

        Args:
            bb (List[int]): The bounding box coordinates [x1, y1, x2, y2].
            face_landmarks (List[List[int]]): The face landmarks coordinates
                                              [[x1, y1], [x2, y2], ...].

        Returns:
            List[dlib.full_object_detection]: A list of dlib.full_object_detection objects.
        """
        rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        dlib_parts = []
        for pt in face_landmarks:
            dlib_parts.append(dlib.point(pt[0], pt[1]))
        return dlib.full_object_detection(rect=rect, parts=dlib_parts)
