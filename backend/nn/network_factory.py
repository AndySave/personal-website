from .network import SmallModel, MediumModel, LargeModel
from .datasets.base_dataset import BaseDataset
from backend.schemas import TaskType, ModelSize

class ModelFactory:
    def create(self, model_size: ModelSize, dataset: BaseDataset, task_type: TaskType):
        if model_size == ModelSize.small:
            return SmallModel(dataset.num_features(), dataset.num_outputs(), task_type)
        elif model_size == ModelSize.medium:
            return MediumModel(dataset.num_features(), dataset.num_outputs(), task_type)
        elif model_size == ModelSize.large:
            return LargeModel(dataset.num_features(), dataset.num_outputs(), task_type)
        else:
            raise ValueError(f"Unknown model size: {model_size}")
