
import numpy as np
import nano_nn as nn

def normalize_activation(values: np.ndarray):
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def get_activations(model: nn.Module, X: np.ndarray) -> list[list[int]]:
    layer_activations = []
    temp = X[:]
    for layer in model.layers:
        temp = layer.forward(temp)

        # We want the raw output from the last layer
        if layer is model.layers[-1]:
            layer_activations.append(temp[0])
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Softmax)):  # TODO: Improve the nn framework so I can use isinstance(layer, nn.Activation) instead
            layer_activations.append(normalize_activation(temp[0]))

    layer_activations = [a.tolist() for a in layer_activations]
    return layer_activations
