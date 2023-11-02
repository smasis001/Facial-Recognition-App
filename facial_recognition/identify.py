"""Identify Faces (using Vector DB) Functionality"""
# pylint: disable=E1101,E0401,C0413
from typing import Tuple, List, Optional
import os
import faiss
import joblib
import numpy as np

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/vectordb/faces_l2.faiss')
VECTOR_KEYS_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/vectordb/faces_l2.pkl')
VECTOR_METRIC = 'euclidean'
VECTOR_DIMENSIONS = 128
FACE_RECOGNITION_TOLERANCE = 0.4

class FaceIdentification:
    """Class for Face Identification"""
    def __init__(
            self,
            tolerance = FACE_RECOGNITION_TOLERANCE
        ) -> None:
        self.tolerance = tolerance
        self.vectordb = faiss.read_index(VECTOR_DB_PATH)
        self.vectorkeys = joblib.load(VECTOR_KEYS_PATH)

    def get_name(
            self,
            i:int
        ) -> str:
        """Get the name of an image in the vector DB based on its sequential ID.

        Args:
            i (int): The sequential ID in the vector DB.

        Returns:
            str: The name of the file as stored in the keys pickle objects.
        """
        path = self.vectorkeys[i]
        if '/' in path:
            name = os.path.basename(os.path.dirname(path))
        else:
            name = path
        return name

    def identify(
            self,
            descriptors:List[np.ndarray],
            k:Optional[int] = 1
        ) -> Tuple[List, List]:
        """Identifies the given descriptors by searching for nearest neighbors in the vector
           database.

        Args:
            descriptors (List[np.ndarray]): A list of numpy arrays representing the descriptors to
                                            be identified.
            k (Optional[int]): The number of nearest neighbors to search for. Defaults to 1.

        Returns:
            Tuple[List, List]: A tuple containing two lists. The first list contains the distances
                               of the nearest neighbors, and the second list contains the names of
                               the nearest neighbors.

        Example:
            distances, neighbors = identify(descriptors, k=3)
        """
        distances, neighbors = self.vectordb.search(np.array(descriptors), k)
        masks = [d < self.tolerance for d in distances]
        distances = [[i for i in d[m]]\
                     for d, m in zip(distances, masks)]
        neighbors = [[self.get_name(i) for i in n[m]]\
                     for n, m in zip(neighbors, masks)]

        return distances, neighbors

    def count(
            self,
            name:str
        ) -> int:
        """Counts the number of occurrences of a given name in the list of names associated with
           vector keys.

        Args:
            name (str): The name to count occurrences for.

        Returns:
            int: The number of occurrences of the given name.
        """
        self.vectorkeys = joblib.load(VECTOR_KEYS_PATH)
        key_cnt = len(self.vectorkeys)
        names = [self.get_name(i) for i in range(key_cnt)]
        return len([n for n in names if n == name])
