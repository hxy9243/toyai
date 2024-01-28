from .engine import Value


def test_basic_value():
    x = Value(3.0)
    y = Value(6.0)

    assert ((x + 3) * 2).val == 12.0
    assert ((3 * x) + 5).val == 14.0
    assert (x * y + y * 2).val == 30.0
    assert (x**2 / y).val == 1.5
    assert (x**2 - y).val == 3.0
