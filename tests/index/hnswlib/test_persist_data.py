import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


class NestedDoc(BaseDoc):
    d: SimpleDoc
    tens: NdArray[50]


def test_persist_and_restore(tmp_path):
    query = SimpleDoc(tens=np.random.random((10,)))

    # create index
    store = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    store.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert store.num_docs() == 10
    find_results_before = store.find(query, search_field='tens', limit=5)

    # delete and restore
    del store
    store = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    assert store.num_docs() == 10
    find_results_after = store.find(query, search_field='tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # add new data
    store.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert store.num_docs() == 15


def test_persist_and_restore_nested(tmp_path):
    query = NestedDoc(
        tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
    )

    # create index
    store = HnswDocumentIndex[NestedDoc](work_dir=str(tmp_path))
    store.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(10)
        ]
    )
    assert store.num_docs() == 10
    find_results_before = store.find(query, search_field='d__tens', limit=5)

    # delete and restore
    del store
    store = HnswDocumentIndex[NestedDoc](work_dir=str(tmp_path))
    assert store.num_docs() == 10
    find_results_after = store.find(query, search_field='d__tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # delete and restore
    store.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(5)
        ]
    )
    assert store.num_docs() == 15


def test_persist_index_file(tmp_path):
    store = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    store = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    store.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert store.num_docs() == 10
