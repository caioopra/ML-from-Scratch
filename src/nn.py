from abc import ABC, abstractmethod

from math import exp
from random import uniform

from typing import Callable

from src.Value import Value


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self): ...

    @abstractmethod
    def __call__(self): ...


class Neuron(Module):
    def __init__(self, n_inputs: int, activation: Callable = None):
        self.w = [Value(uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation is None:
            return out

        return self.activation(out)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int, activation: Callable = None):
        self.neurons = [
            Neuron(n_inputs, activation=activation) for _ in range(n_outputs)
        ]
        self.activation = activation

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


def relu(input: list, derivative: bool = False) -> list:
    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            new = Value(max(0, x.data), (x,), "relu")

            def _backward():
                new.grad += 1 if x.data > 0 else 0

            new._backward = _backward

            new_nodes.append(new)

        return new_nodes

    if derivative:
        return [1 if x > 0 else 0 for x in input]

    return [max(0, x) for x in input]


# TODO: add treatment for when is not Value
def tanh(input: list | Value) -> list | Value:
    if isinstance(input, Value):
        x = input.data
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1)
        new = Value(t, (input,), "tanh")

        def _backward():
            input.grad += (1 - t**2) * new.grad

        new._backward = _backward
        return new

    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            data = x.data
            t = (exp(2 * data) - 1) / (exp(2 * data) + 1)
            new = Value(t, (x,), "tanh")

            def _backward():
                new.grad += (1 - t**2) * t.grad

            new._backward = _backward

            new_nodes.append(new)

        return new_nodes


# TODO: implement
def sigmoid(input: list) -> list: ...
