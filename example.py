import math

import numpy as np

from toygrad.nn import Module, Linear

STEPS = 10


class Model(Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.layer1 = Linear(input_features, 8)
        self.layer2 = Linear(8, 8)
        self.layer3 = Linear(8, 4)
        self.output = Linear(4, output_features)

    def forward(self, X):
        X = self.layer1(X)
        X = [xi.relu() for xi in X]
        X = self.layer2(X)
        X = [xi.relu() for xi in X]
        X = self.layer3(X)
        X = [xi.relu() for xi in X]
        output = self.output(X)
        return output

    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.layer3.parameters()
            + self.output.parameters()
        )

    def train(self, X, y, epochs=1, learning_rate=1e-5):
        for epoch in range(epochs):
            loss = 0.0

            for i, xval in enumerate(X):
                out = self.forward(xval)

                # get cost function
                cost = ((y[i] - out) ** 2)[0]

                # backward
                cost.backward()

                # apply the grad with learning rate
                for p in self.parameters():
                    p.val -= p.grad * learning_rate

                # zero out the grads
                for p in self.parameters():
                    p.grad = 0.0

                loss += cost.val**0.5

            loss = loss / len(X)
            print(f"Epoch {epoch+1}: loss {loss}")


def main():
    X = np.linspace(-2 * math.pi, 6 * math.pi)
    y = np.sin(X) ** 2 + 1

    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    print(X[0])
    print(y[0])

    m = Model(1, 1)
    m.train(X, y, epochs=60)


if __name__ == "__main__":
    main()
