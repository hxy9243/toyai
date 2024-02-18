ToyGrad
====

A toy implementation of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), inspired by A. Karparthy's https://github.com/karpathy/micrograd

# Basic Concepts

See more about Autograd's basic concpets, architecture, and implementation at the [basic concept of automatic differentiation](autograd.md).

# Demo

See demo in Jupyter [notebook how to use ToyGrad](demo.ipynb).

```python
class Model(Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.layer1 = Linear(input_features, 8)
        self.layer2 = Linear(8, 4)
        self.output = Linear(4, output_features)

    def forward(self, X):
        X = self.layer1(X)
        X = [xi.relu() for xi in X]
        X = self.layer2(X)
        X = [xi.relu() for xi in X]
        output = self.output(X)
        return output
```

![](input_predict_chart.png)