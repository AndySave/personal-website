
from typing import Literal
import nano_nn as nn
from schemas import Layer, LayerType, NetworkMetadata

class SmallModel(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, task_type: Literal["regression", "binary_classification", "multi_classification"]):
        super().__init__()

        self.add(nn.Dense(num_features, 4))
        self.add(nn.ReLU())
        self.add(nn.Dense(4, 2))
        self.add(nn.ReLU())
        self.add(nn.Dense(2, num_outputs))

        match (task_type):
            case "regression":
                self.set_loss_fn(nn.MeanSquaredError())
            case "binary_classification":
                self.add(nn.Sigmoid())
                self.set_loss_fn(nn.BinaryCrossEntropy())
            case "multi_classification":
                self.add(nn.Softmax())
                self.set_loss_fn(nn.CategoricalCrossEntropy())
        
        self.set_learning_rate(0.01)
        self.set_optimizer(nn.Adam())
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Small Network", model_size="small", layer_sizes=[4, 5])


class MediumModel(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, task_type: Literal["regression", "binary_classification", "multi_classification"]):
        super().__init__()

        self.add(nn.Dense(num_features, 8))
        self.add(nn.ReLU())
        self.add(nn.Dense(8, 16))
        self.add(nn.ReLU())
        self.add(nn.Dense(16, 6))
        self.add(nn.ReLU())
        self.add(nn.Dense(6, num_outputs))

        match (task_type):
            case "regression":
                self.set_loss_fn(nn.MeanSquaredError())
            case "binary_classification":
                self.add(nn.Sigmoid())
                self.set_loss_fn(nn.BinaryCrossEntropy())
            case "multi_classification":
                self.add(nn.Softmax())
                self.set_loss_fn(nn.CategoricalCrossEntropy())
        
        self.set_learning_rate(0.01)
        self.set_optimizer(nn.Adam())
    
    @staticmethod
    def metadata() -> NetworkMetadata:
        return NetworkMetadata(display_name="Medium Network", model_size="medium", layer_sizes=[4, 5, 8, 4, 1])
