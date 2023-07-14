import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.milvus.fixtures import start_storage, tmp_index_name  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(is_embedding=True)


class FlatDoc(BaseDoc):
    tens_one: NdArray[10] = Field(is_embedding=True)
    tens_two: NdArray[50]


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class TorchDoc(BaseDoc):
    tens: TorchTensor[10] = Field(is_embedding=True)  # type: ignore[valid-type]


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


@pytest.fixture
def ten_flat_docs():
    return [
        FlatDoc(tens_one=np.random.randn(10), tens_two=np.random.randn(50))
        for _ in range(10)
    ]


@pytest.fixture
def ten_nested_docs():
    return [NestedDoc(d=SimpleDoc(tens=np.random.randn(10))) for _ in range(10)]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(
    ten_simple_docs, use_docarray, tmp_index_name
):  # noqa: F811
    index = MilvusDocumentIndex[SimpleDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_simple_docs = DocList[SimpleDoc](ten_simple_docs)

    index.index(ten_simple_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, use_docarray, tmp_index_name):  # noqa: F811
    index = MilvusDocumentIndex[FlatDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_flat_docs = DocList[FlatDoc](ten_flat_docs)

    index.index(ten_flat_docs)
    assert index.num_docs() == 10


def test_index_torch(tmp_index_name):
    docs = [TorchDoc(tens=np.random.randn(10)) for _ in range(10)]
    assert isinstance(docs[0].tens, torch.Tensor)
    assert isinstance(docs[0].tens, TorchTensor)

    index = MilvusDocumentIndex[TorchDoc](index_name=tmp_index_name)

    index.index(docs)
    assert index.num_docs() == 10


def test_del_single(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = MilvusDocumentIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)
    # delete once
    assert index.num_docs() == 10
    del index[ten_simple_docs[0].id]
    assert index.num_docs() == 9
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i == 0:  # deleted
            with pytest.raises(KeyError):
                index[id_]
        else:
            assert index[id_].id == id_
    # delete again
    del index[ten_simple_docs[3].id]
    assert index.num_docs() == 8
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i in (0, 3):  # deleted
            with pytest.raises(KeyError):
                index[id_]
        else:
            assert index[id_].id == id_


def test_del_multiple(ten_simple_docs, tmp_index_name):
    docs_to_del_idx = [0, 2, 4, 6, 8]

    index = MilvusDocumentIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_del = [ten_simple_docs[i] for i in docs_to_del_idx]
    ids_to_del = [d.id for d in docs_to_del]
    del index[ids_to_del]
    for i, doc in enumerate(ten_simple_docs):
        if i in docs_to_del_idx:
            with pytest.raises(KeyError):
                index[doc.id]
        else:
            assert index[doc.id].id == doc.id


def test_num_docs(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = MilvusDocumentIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10

    del index[ten_simple_docs[0].id]
    assert index.num_docs() == 9

    del index[ten_simple_docs[3].id, ten_simple_docs[5].id]
    assert index.num_docs() == 7

    more_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(5)]
    index.index(more_docs)
    assert index.num_docs() == 12

    del index[more_docs[2].id, ten_simple_docs[7].id]  # type: ignore[arg-type]
    assert index.num_docs() == 10
