
from abc import ABC, abstractmethod
from schemas import CustomInput

class BaseDataset(ABC):
    @abstractmethod
    def _get_dataset(self): ...

    @abstractmethod
    def transform_one(self, custom_input: CustomInput): ...
    
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def num_classes(self) -> int: ...

    @abstractmethod
    def metadata(self) -> dict: ...