import pytest

from docarray.array.array.typed_sequence import TypedList


@pytest.fixture()
def data():
    class A:
        def __init__(self, param=None):
            self.param = param

        def f(self):
            return None

        def g(self):
            return 'is this really working ?'

    return TypedList[A]([A(param=1) for _ in range(3)])


def test_func_call(data):

    assert data.f() == [None, None, None]

    assert data.g() == ['is this really working ?'] * 3


def test_atr_access(data):
    assert data.param == [1, 1, 1]
