import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import start_storage, tmp_collection_name  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]


class NestedDoc(BaseDoc):
    d: SimpleDoc
    tens: NdArray[50]  # type: ignore[valid-type]


def test_persist_and_restore(tmp_collection_name):  # noqa: F811
    query = SimpleDoc(tens=np.random.random((10,)))

    # create index
    index = QdrantDocumentIndex[SimpleDoc](
        host='localhost', collection_name=tmp_collection_name
    )
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert index.num_docs() == 10
    find_results_before = index.find(query, search_field='tens', limit=5)

    # delete and restore
    del index
    index = QdrantDocumentIndex[SimpleDoc](
        host='localhost', collection_name=tmp_collection_name
    )
    assert index.num_docs() == 10
    find_results_after = index.find(query, search_field='tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert doc_before.tens == pytest.approx(doc_after.tens)

    # add new data
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert index.num_docs() == 15


def test_persist_and_restore_nested(tmp_collection_name):  # noqa: F811
    query = NestedDoc(
        tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
    )

    # create index
    index = QdrantDocumentIndex[NestedDoc](
        host='localhost', collection_name=tmp_collection_name
    )
    index.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(10)
        ]
    )
    assert index.num_docs() == 10
    find_results_before = index.find(query, search_field='d__tens', limit=5)

    # delete and restore
    del index
    index = QdrantDocumentIndex[NestedDoc](
        host='localhost', collection_name=tmp_collection_name
    )
    assert index.num_docs() == 10
    find_results_after = index.find(query, search_field='d__tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert doc_before.tens == pytest.approx(doc_after.tens)

    # delete and restore
    index.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(5)
        ]
    )
    assert index.num_docs() == 15
