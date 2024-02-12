import math

import numpy as np

from toygrad.nn import Module, Linear

STEPS = 10


class Model(Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.layer1 = Linear(input_features, 8)
        self.layer2 = Linear(8, 8)
        self.output = Linear(8, output_features)

    def forward(self, X):
        X = self.layer1(X)
        X = [xi.relu for xi in X]
        X = self.layer2(X)

        X = [xi.relu for xi in X]
        output = self.output(X)

        return output

    def train(self, X, y, epochs=1, learning_rate=1e-5):
        X_shape = len(X[1])
        y_shape = len(y[1])

        for epoch in range(10):
            for i, xval in enumerate(X):
                out = self.forward(xval)

                # get cost function
                cost = (y - out) ** 2

                # backward
                cost.backward()

                # apply the grad with learning rate
                for p in m.parameters():
                    p.val -= p.grad * learning_rate

                # zero out the grads
                for p in m.parameters():
                    p.grad = 0.0

                print(f"Epoch {epoch} step {i}/{len(X)}: loss {cost ** 0.5}")


def main():
    X = np.linspace(-3 * math.pi, 3 * math.pi)
    y = math.sin(X)

    m = Model(1, 1)
    m.train(X, y)


if __name__ == "__main__":
    main()
