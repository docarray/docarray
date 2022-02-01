import pytest

from docarray import DocumentArray
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate


@pytest.mark.parametrize(
    'da_cls', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
def test_empty_non_zero(da_cls, start_weaviate):
    da = DocumentArray.empty(10)
    assert len(da) == 10
    da = DocumentArray.empty()
    assert len(da) == 0
