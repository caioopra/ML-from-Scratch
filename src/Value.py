from __future__ import annotations


class Value:
    """Stores a single value and it's gradient"""

    def __init__(
        self, data: int | float, _children: tuple[str] = (), _operation: str = ""
    ):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._operation = _operation

    def __add__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(other, (int, float)), "Exponent must be a integer or float"

        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: int | float | Value) -> Value:
        return self + other

    def __rsub__(self, other: int | float | Value) -> Value:
        return other + (-self)

    def __rmul__(self, other: int | float | Value) -> Value:
        return self * other

    def __truediv__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other: int | float | Value) -> Value:
        return other * (self**-1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
