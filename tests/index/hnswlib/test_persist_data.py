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
    _ = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))

    # load existing index file
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    assert index.num_docs() == 0
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert index.num_docs() == 10
    find_results_before = index.find(query, search_field='tens', limit=5)

    # delete and restore
    del index
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    assert index.num_docs() == 10
    find_results_after = index.find(query, search_field='tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert np.allclose(doc_before.tens == doc_after.tens)

    # add new data
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert index.num_docs() == 15


def test_persist_and_restore_nested(tmp_path):
    query = NestedDoc(
        tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
    )

    # create index
    index = HnswDocumentIndex[NestedDoc](work_dir=str(tmp_path))
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
    index = HnswDocumentIndex[NestedDoc](work_dir=str(tmp_path))
    assert index.num_docs() == 10
    find_results_after = index.find(query, search_field='d__tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert np.allclose(doc_before.tens == doc_after.tens)

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
