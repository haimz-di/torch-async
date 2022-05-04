from abc import ABC, abstractmethod

from numpy import ndarray
from typing import List, Tuple


class ChunkDataloader(ABC):
    @abstractmethod
    def load_chunk(self, index: int) -> Tuple[int, ndarray, ndarray]:
        """
        Loads a chunk of data in CPU memory and returns a tuple of (number of samples, features array, labels array).
        :param index: index of the chunk to load
        :return: number of samples, features array, labels array
        """
        pass

    @abstractmethod
    def sample_size(self) -> int:
        """
        Get the size of an individual samples.
        :return: sample size
        """
        pass

    @abstractmethod
    def num_items(self) -> List[int]:
        """
        Get the list of number of samples per chunk.
        :return: list of chunk sizes
        """
        pass

    @abstractmethod
    def num_chunks(self) -> int:
        """
        Get the total number of chunks in the dataset.
        :return: number of chunks
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        :return: size of the dataset
        """
        pass
