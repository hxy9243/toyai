Explanation of the AutoGrad Algorithm
=====

# Key Concepts

## Differentiation

In calculus, differentiation is the process of finding the derivative of
a function. It finds the "rate of change" at any point on the function curve.

Intuitively, for example in Physics, the differentiation of the function of position
of an object is its velocity. The differentiation of the speed of the object is acceration.

## Derivative

Derivate shows the "rate of change" at any point of the function.

## Chain Rule

In differentiation, the chain rule shows the basic rules for finding the derivative
of the composite functions.

It states:

$$ \frac{df(g(x))}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} $$

For example, the derivative of the function $f(x) = sin x^2$ is:

$$ \frac{dsin(x^2)}{dx} = \frac{dsin(x^2)}{dx} \cdot \frac{dx^2}{x} = cos(x^2) \cdot 2x $$

With chain-rule, we could derive the auto-grad algorithm to implement backpropagation
on complex compute graphs.

## Backpropagation

Backpropagation, or backward propagation of errors, is the algorithm to find the derivative
of the model (which is a function from inputs to output).
It calcuates the gradient backwards through the feed-foward network from the
last layer to the first.

## AutoGrad

AutoGrad, or automatic differentiation, is the implementation of the backpropagation algorithm.
It applies the gradient by chain rule in a reverse manner, calculating the derivative of all
the function inputs by applying the gradient backwards in the compute graph.

It's the core of training any ML model with Gradient Descent. It applies the backpropagation
on the loss of the function and the training data, to find the optimal model parameters.

https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd

![Autograd](https://raw.githubusercontent.com/karpathy/micrograd/c911406e5ace8742e5841a7e0df113ecb5d54685/gout.svg)

## Gradient Descent

Gradient Descent is an optimization algorithm that typically involves the three steps:

- With a pre-defined model, compute the difference (loss) between the result and the actual output from training data.
- By using backpropagation, compute the gradient of all the model parameters backward,
  apply the loss to update the parameters.
- Repeat, until the loss is minimized.

Think of it as a way of slowly approaching the optimal point based on the loss with the model function.

![Gradient Descent](https://upload.wikimedia.org/wikipedia/commons/f/ff/Gradient_descent.svg)

# References

- https://www.britannica.com/science/differentiation-mathematics
- https://en.wikipedia.org/wiki/Derivative
- https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/
- https://github.com/karpathy/micrograd
- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- https://deepai.org/machine-learning-glossary-and-terms/backpropagation