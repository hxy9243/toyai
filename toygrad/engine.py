class Value:
    def __init__(self, val):
        self.val = val
        self.grad = 0
        self.operands = []

    def __repr__(self):
        return f"<Value {self.val}>"

    def __eq__(self, other):
        other = Value(other) if not isinstance(other, Value) else other

        return self.val == other.val

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val + other.val)
        v.operands = [self, other]
        return v

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val * other.val)
        v.operands = [self, other]
        return v

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val**other.val)
        v.operands = [self, other]
        return v

    def __truediv__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        v = Value(self.val / other.val)
        v.operands = [self, other]
        return v

    def __rtruediv__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        return other.__truediv__(self)
