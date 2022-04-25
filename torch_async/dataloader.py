from abc import ABC, abstractmethod

from numpy import ndarray
from typing import List, Tuple


class ChunkDataloader(ABC):
    @abstractmethod
    def load_chunk(self, index: int) -> Tuple[int, ndarray, ndarray]:
        pass

    @abstractmethod
    def sample_size(self) -> int:
        pass

    @abstractmethod
    def num_items(self) -> List[int]:
        pass

    @abstractmethod
    def num_chunks(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
