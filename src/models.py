from typing import Callable

from src.nn import Module, Neuron, Layer, Value


class MLP(Module):
    """Multi Layer Perceptron - fully connected neural network"""

    def __init__(self, n_inputs: int, n_outputs: list, activation: Callable = None):
        sz = [n_inputs] + n_outputs

        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activation)
            for i in range(len(n_outputs))
        ]
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class LinearRegression(Module):
    def __init__(self):
        self.neuron = Neuron(1)

    def __call__(self, x) -> Value:
        return self.neuron(x)

    def parameters(self):
        return self.neuron.parameters()
