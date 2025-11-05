
import nano_nn as nn
from schemas import NetworkMetadata, TaskType

class SmallModel(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, task_type: TaskType):
        super().__init__()

        self.add(nn.Dense(num_features, 4))
        self.add(nn.ReLU())
        self.add(nn.Dense(4, 2))
        self.add(nn.ReLU())
        self.add(nn.Dense(2, num_outputs))

        match (task_type):
            case TaskType.regression:
                self.set_loss_fn(nn.MeanSquaredError())
            case TaskType.binary_classification:
                self.add(nn.Sigmoid())
                self.set_loss_fn(nn.BinaryCrossEntropy())
            case TaskType.multi_classification:
                self.add(nn.Softmax())
                self.set_loss_fn(nn.CategoricalCrossEntropy())
        
        self.set_learning_rate(0.01)
        self.set_optimizer(nn.Adam())
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Small Network", model_size="small")


class MediumModel(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, task_type: TaskType):
        super().__init__()

        self.add(nn.Dense(num_features, 6))
        self.add(nn.ReLU())
        self.add(nn.Dense(6, 8))
        self.add(nn.ReLU())
        self.add(nn.Dense(8, 6))
        self.add(nn.ReLU())
        self.add(nn.Dense(6, num_outputs))

        match (task_type):
            case TaskType.regression:
                self.set_loss_fn(nn.MeanSquaredError())
            case TaskType.binary_classification:
                self.add(nn.Sigmoid())
                self.set_loss_fn(nn.BinaryCrossEntropy())
            case TaskType.multi_classification:
                self.add(nn.Softmax())
                self.set_loss_fn(nn.CategoricalCrossEntropy())
        
        self.set_learning_rate(0.01)
        self.set_optimizer(nn.Adam())
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Medium Network", model_size="medium")


class LargeModel(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, task_type: TaskType):
        super().__init__()

        self.add(nn.Dense(num_features, 12))
        self.add(nn.ReLU())
        self.add(nn.Dense(12, 16))
        self.add(nn.ReLU())
        self.add(nn.Dense(16, 16))
        self.add(nn.ReLU())
        self.add(nn.Dense(16, 12))
        self.add(nn.ReLU())
        self.add(nn.Dense(12, num_outputs))

        match (task_type):
            case TaskType.regression:
                self.set_loss_fn(nn.MeanSquaredError())
            case TaskType.binary_classification:
                self.add(nn.Sigmoid())
                self.set_loss_fn(nn.BinaryCrossEntropy())
            case TaskType.multi_classification:
                self.add(nn.Softmax())
                self.set_loss_fn(nn.CategoricalCrossEntropy())
        
        self.set_learning_rate(0.01)
        self.set_optimizer(nn.Adam())
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Large Network", model_size="large")
