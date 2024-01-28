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
