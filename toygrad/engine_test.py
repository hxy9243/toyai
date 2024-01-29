from .engine import Value


def test_basic_value():
    x = Value(3.0)
    y = Value(6.0)

    assert ((x + 3) * 2).val == 12.0
    assert ((3 * x) + 5).val == 14.0
    assert (x * y + y * 2).val == 30.0
    assert (x**2 / y).val == 1.5
    assert (x**2 - y).val == 3.0


def test_grad1():
    x = Value(3.0)
    y = Value(7.0)
    z = Value(2.0)

    v = (x + y) * z
    v.backward()
    assert v.val == 20.0

    assert x.grad == 2.0
    assert y.grad == 2.0
    assert z.grad == 10.0


def test_grad2():
    x = Value(3.0)
    y = Value(7.0)
    z = Value(2.0)
    w = Value(5.0)

    v = (x + y) * z * w**2
    v.backward()
    assert v.val == 500.0

    assert x.grad == 50.0
    assert y.grad == 50.0
    assert z.grad == 250.0
    assert w.grad == 200.0
