
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def _get_dataset(self): ...

    @abstractmethod
    def transform_one(self): ...
    
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def num_classes(self) -> int: ...

    @abstractmethod
    def metadata(self) -> dict: ...
