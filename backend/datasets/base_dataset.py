
from abc import ABC, abstractmethod
from typing import Literal
from schemas import DatasetMetadata

class BaseDataset(ABC):
    @abstractmethod
    def _get_dataset(self): ...

    @abstractmethod
    def transform_one(self): ...
    
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def num_outputs(self) -> int: ...

    @abstractmethod
    def task_type(self) -> Literal["regression", "binary_classification", "multi_classification"]: ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...
