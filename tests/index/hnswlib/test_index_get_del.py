import os
from typing import Optional

import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc, TextDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray, NdArrayEmbedding, TorchTensor

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class TorchDoc(BaseDoc):
    tens: TorchTensor[10]


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
def test_index_simple_schema(ten_simple_docs, tmp_path, use_docarray):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_simple_docs = DocList[SimpleDoc](ten_simple_docs)

    index.index(ten_simple_docs)
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


def test_schema_with_user_defined_mapping(tmp_path):
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, col_type=np.ndarray)

    index = HnswDocumentIndex[MyDoc](work_dir=str(tmp_path))
    assert index._column_infos['tens'].db_type == np.ndarray


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, tmp_path, use_docarray):
    index = HnswDocumentIndex[FlatDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_flat_docs = DocList[FlatDoc](ten_flat_docs)

    index.index(ten_flat_docs)
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, tmp_path, use_docarray):
    index = HnswDocumentIndex[NestedDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_nested_docs = DocList[NestedDoc](ten_nested_docs)

    index.index(ten_nested_docs)
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


def test_index_torch(tmp_path):
    docs = [TorchDoc(tens=np.random.randn(10)) for _ in range(10)]
    assert isinstance(docs[0].tens, torch.Tensor)
    assert isinstance(docs[0].tens, TorchTensor)

    index = HnswDocumentIndex[TorchDoc](work_dir=str(tmp_path))

    index.index(docs)
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


@pytest.mark.tensorflow
def test_index_tf(tmp_path):
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]

    docs = [TfDoc(tens=np.random.randn(10)) for _ in range(10)]
    # assert isinstance(docs[0].tens, torch.Tensor)
    assert isinstance(docs[0].tens, TensorFlowTensor)

    index = HnswDocumentIndex[TfDoc](work_dir=str(tmp_path))

    index.index(docs)
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


def test_index_builtin_docs(tmp_path):
    # TextDoc
    class TextSchema(TextDoc):
        embedding: Optional[NdArrayEmbedding] = Field(dim=10)

    index = HnswDocumentIndex[TextSchema](work_dir=str(tmp_path))

    index.index(
        DocList[TextDoc](
            [TextDoc(embedding=np.random.randn(10), text=f'{i}') for i in range(10)]
        )
    )
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10

    # ImageDoc
    class ImageSchema(ImageDoc):
        embedding: Optional[NdArrayEmbedding] = Field(dim=10)

    index = HnswDocumentIndex[ImageSchema](
        work_dir=str(os.path.join(tmp_path, 'image'))
    )

    index.index(
        DocList[ImageDoc](
            [
                ImageDoc(
                    embedding=np.random.randn(10), tensor=np.random.randn(3, 224, 224)
                )
                for _ in range(10)
            ]
        )
    )
    assert index.num_docs() == 10
    for index in index._hnsw_indices.values():
        assert index.get_current_count() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    simple_path = tmp_path / 'simple'
    flat_path = tmp_path / 'flat'
    nested_path = tmp_path / 'nested'

    # simple
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(simple_path))
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens == d.tens)

    # flat
    index = HnswDocumentIndex[FlatDoc](work_dir=str(flat_path))
    index.index(ten_flat_docs)

    assert index.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens_one == d.tens_one)
        assert np.all(index[id_].tens_two == d.tens_two)

    # nested
    index = HnswDocumentIndex[NestedDoc](work_dir=str(nested_path))
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert index[id_].d.id == d.d.id
        assert np.all(index[id_].d.tens == d.d.tens)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    simple_path = tmp_path / 'simple'
    flat_path = tmp_path / 'flat'
    nested_path = tmp_path / 'nested'
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(simple_path))
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    index = HnswDocumentIndex[FlatDoc](work_dir=str(flat_path))
    index.index(ten_flat_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_flat_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens_one == d_in.tens_one)
        assert np.all(d_out.tens_two == d_in.tens_two)

    # nested
    index = HnswDocumentIndex[NestedDoc](work_dir=str(nested_path))
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)


def test_get_key_error(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    index.index(ten_simple_docs)

    with pytest.raises(KeyError):
        index['not_a_real_id']


def test_del_single(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
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
            assert np.all(index[id_].tens == d.tens)
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
            assert np.all(index[id_].tens == d.tens)


def test_del_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    docs_to_del_idx = [0, 2, 4, 6, 8]

    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
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
            assert np.all(index[doc.id].tens == doc.tens)


def test_del_key_error(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    index.index(ten_simple_docs)

    with pytest.raises(KeyError):
        del index['not_a_real_id']


def test_num_docs(ten_simple_docs, tmp_path):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    index.index(ten_simple_docs)

    assert index.num_docs() == 10

    del index[ten_simple_docs[0].id]
    assert index.num_docs() == 9

    del index[ten_simple_docs[3].id, ten_simple_docs[5].id]
    assert index.num_docs() == 7

    more_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(5)]
    index.index(more_docs)
    assert index.num_docs() == 12

    del index[more_docs[2].id, ten_simple_docs[7].id]
    assert index.num_docs() == 10
