
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

from backend.nn_playground.schemas import DatasetMetadata, TaskType

class BaseDataset(ABC):
    X: Any
    y: Any

    @abstractmethod
    def _get_dataset(self): ...

    @abstractmethod
    def transform_one(self, input: BaseModel) -> ndarray: ...
    
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def num_outputs(self) -> int: ...

    @abstractmethod
    def task_type(self) -> TaskType: ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...
