from typing import List
from random import random

from toygrad.engine import Value


class Module:
    def zero_grad(self):
        pass

    def forward(self):
        pass

    def parameters(self):
        pass


class Linear:
    def __init__(self, in_features: int, out_features: int):
        """Creates a linear layer of neurals"""

        self.in_features = in_features
        self.out_features = out_features

        self.weights = [Value(random()) for _ in range(out_features)]
        self.bias = Value(random())

    def __call__(self, in_values: List[Value]) -> List[Value]:
        out_values = []

        for p in self.parameters:
            out = Value(0.0)
            for v in in_values:
                out += p * v

            out_values.append(out + self.bias)

        return out_values

    def parameters(self):
        params = [w for w in self.weights]
        params.append(self.bias)

        return params
