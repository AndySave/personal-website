
import nano_nn as nn

from .datasets import BaseDataset


class InMemoryRepository:
    def __init__(self):
        self.models: dict[str, nn.Module] = {}
        self.datasets: dict[str, BaseDataset] = {}
    
    def add_model(self, model_id: str, model: nn.Module) -> None:
        self.models[model_id] = model
    
    def get_model(self, model_id: str) -> nn.Module:
        return self.models[model_id]
    
    def add_dataset(self, dataset_name: str, dataset: BaseDataset) -> None:
        self.datasets[dataset_name] = dataset

    def get_dataset(self, dataset_name: str) -> BaseDataset:
        return self.datasets[dataset_name]
