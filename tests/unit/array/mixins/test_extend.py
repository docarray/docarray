import pytest
from docarray import DocumentArray


def test_extend():
    da = DocumentArray.empty(3)
    assert len(da) == 3

    with pytest.raises(AttributeError):
        da.extend([1, 2, 3])

    assert len(da) == 3
    da.summary()
