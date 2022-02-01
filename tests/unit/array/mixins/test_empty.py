import pytest

from docarray import DocumentArray
from docarray.array.sqlite import DocumentArraySqlite


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArraySqlite])
def test_empty_non_zero(da_cls):
    da = DocumentArray.empty(10)
    assert len(da) == 10
    da = DocumentArray.empty()
    assert len(da) == 0
