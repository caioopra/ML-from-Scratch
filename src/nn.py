from random import uniform
from src.Value import Value

from abc import ABC, abstractmethod


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self): ...

    @abstractmethod
    def __call__(self): ...


# TODO: add non linearity
class Neuron(Module):
    def __init__(self, n_inputs: int):
        self.w = [Value(uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0)

    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

