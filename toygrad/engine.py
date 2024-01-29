class Value:
    def __init__(self, val):
        self.val = val
        self.grad = 0
        self.operands = set()
        self._backward = lambda: None

    def __repr__(self):
        return f"<Value {self.val} with grad {self.grad}>"

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val + other.val)
        v.operands = set((self, other))

        def _backward():
            self.grad += v.grad
            other.grad += v.grad

        v._backward = _backward
        return v

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val * other.val)
        v.operands = set((self, other))

        def _backward():
            self.grad += other.val * v.grad
            other.grad += self.val * v.grad

        v._backward = _backward

        return v

    def relu(self):
        if self.val > 0:
            v = Value(self.val)
        else:
            v = Value(0.0)
        v.operands = set((self,))

        def _backward():
            if v.val > 0:
                self.grad = v.grad
            else:
                self.grad = 0

        v._backward = _backward

        return v

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val**other.val)
        v.operands = set((self,))

        def _backward():
            self.grad += other.val * self.val ** (other.val - 1) * v.grad

        v._backward = _backward
        return v

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        # propagate grad backward to the previous operands
        # do a reverse topo sort to get the order of backward propagation
        visited = set()
        results = []

        def topo(v):
            for op in v.operands:
                if op in visited:
                    continue
                visited.add(op)
                topo(op)
            results.append(v)

        topo(self)
        self.grad = 1

        for v in reversed(results):
            v._backward()
