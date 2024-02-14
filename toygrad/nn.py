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

        # create a matrix of shape out_features * in_features
        # output y = Wx + b
        self.weights = [
            [Value(random()) for _ in range(in_features)] for _ in range(out_features)
        ]
        self.bias = [Value(random()) for _ in range(out_features)]

    def __call__(self, in_values: List[Value]) -> List[Value]:
        if len(in_values) != self.in_features:
            raise ValueError(
                f"Expecting input size {self.in_features}, getting {len(in_values)}"
            )

        out_values = []
        for i in range(self.out_features):
            out = Value(0.0)
            for j in range(self.in_features):
                out += self.weights[i][j] * in_values[j]

            out_values.append(out)

        for i in range(self.out_features):
            out_values[i] += self.bias[i]

        return out_values

    def parameters(self):
        return [w for lines in self.weights for w in lines] + [b for b in self.bias]
