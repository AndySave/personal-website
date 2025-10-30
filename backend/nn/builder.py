
import nano_nn as nn
from schemas import Layer, LayerType

def add_layer(module: nn.Module, layer: Layer):
    match layer.type:
        case LayerType.dense:
            module.add(nn.Dense(layer.in_features, layer.out_features))
        case LayerType.dropout:
            module.add(nn.Dropout(0.5))  # TODO: Change later to accept custom rate
        case LayerType.relu:
            module.add(nn.ReLU())
        case LayerType.sigmoid:
            module.add(nn.Sigmoid())
        case LayerType.softmax:
            module.add(nn.Softmax())


def build_model(layers: list[Layer]):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            for layer in layers:
                add_layer(self, layer)

    model = Model()
    model.set_learning_rate(0.01)
    model.set_loss_fn(nn.BinaryCrossEntropy())
    model.set_optimizer(nn.Adam())

    return model
