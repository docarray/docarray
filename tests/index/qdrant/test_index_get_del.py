from typing import Optional

import numpy as np
import pytest
import torch
from pydantic import Field
from scipy.spatial.distance import cosine

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc, TextDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray, NdArrayEmbedding, TorchTensor

import qdrant_client

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
def qdrant_config():
    return QdrantDocumentIndex.DBConfig(
        url='http://localhost:6333',
    )


@pytest.fixture
def qdrant():
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient('http://localhost:6333')
    client.delete_collection(collection_name='documents')


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
def test_index_simple_schema(ten_simple_docs, qdrant_config, use_docarray, qdrant):
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    if use_docarray:
        ten_simple_docs = DocList[SimpleDoc](ten_simple_docs)

    store.index(ten_simple_docs)
    assert store.num_docs() == 10


def test_schema_with_user_defined_mapping(qdrant_config, qdrant):
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, col_type=np.ndarray)

    store = QdrantDocumentIndex[MyDoc](db_config=qdrant_config)
    assert store._column_infos['tens'].db_type == np.ndarray


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, qdrant_config, use_docarray, qdrant):
    store = QdrantDocumentIndex[FlatDoc](db_config=qdrant_config)
    if use_docarray:
        ten_flat_docs = DocList[FlatDoc](ten_flat_docs)

    store.index(ten_flat_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, qdrant_config, use_docarray, qdrant):
    store = QdrantDocumentIndex[NestedDoc](db_config=qdrant_config)
    if use_docarray:
        ten_nested_docs = DocList[NestedDoc](ten_nested_docs)

    store.index(ten_nested_docs)
    assert store.num_docs() == 10


def test_index_torch(qdrant_config, qdrant):
    docs = [TorchDoc(tens=np.random.randn(10)) for _ in range(10)]
    assert isinstance(docs[0].tens, torch.Tensor)
    assert isinstance(docs[0].tens, TorchTensor)

    store = QdrantDocumentIndex[TorchDoc](db_config=qdrant_config)

    store.index(docs)
    assert store.num_docs() == 10


@pytest.mark.skip('Qdrant does not support storing image tensors yet')
def test_index_builtin_docs(qdrant_config, qdrant):
    # TextDoc
    class TextSchema(TextDoc):
        embedding: Optional[NdArrayEmbedding] = Field(dim=10)

    store = QdrantDocumentIndex[TextSchema](db_config=qdrant_config)

    store.index(
        DocList[TextDoc](
            [TextDoc(embedding=np.random.randn(10), text=f'{i}') for i in range(10)]
        )
    )
    assert store.num_docs() == 10

    # ImageDoc
    class ImageSchema(ImageDoc):
        embedding: Optional[NdArrayEmbedding] = Field(dim=10)

    store = QdrantDocumentIndex[ImageSchema](collection_name='images')

    store.index(
        DocList[ImageDoc](
            [
                ImageDoc(
                    embedding=np.random.randn(10), tensor=np.random.randn(3, 224, 224)
                )
                for _ in range(10)
            ]
        )
    )
    assert store.num_docs() == 10


def test_get_key_error(ten_simple_docs, ten_flat_docs, ten_nested_docs, qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index(ten_simple_docs)

    with pytest.raises(KeyError):
        store['not_a_real_id']


def test_del_single(ten_simple_docs, ten_flat_docs, ten_nested_docs, qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index(ten_simple_docs)
    # delete once
    assert store.num_docs() == 10
    del store[ten_simple_docs[0].id]
    assert store.num_docs() == 9
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i == 0:  # deleted
            with pytest.raises(KeyError):
                store[id_]
        else:
            assert store[id_].id == id_
    # delete again
    del store[ten_simple_docs[3].id]
    assert store.num_docs() == 8
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i in (0, 3):  # deleted
            with pytest.raises(KeyError):
                store[id_]
        else:
            assert store[id_].id == id_


def test_del_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs, qdrant_config, qdrant):
    docs_to_del_idx = [0, 2, 4, 6, 8]

    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    docs_to_del = [ten_simple_docs[i] for i in docs_to_del_idx]
    ids_to_del = [d.id for d in docs_to_del]
    del store[ids_to_del]
    for i, doc in enumerate(ten_simple_docs):
        if i in docs_to_del_idx:
            with pytest.raises(KeyError):
                store[doc.id]
        else:
            assert store[doc.id].id == doc.id


def test_del_key_error(ten_simple_docs, ten_flat_docs, ten_nested_docs, qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index(ten_simple_docs)

    with pytest.raises(KeyError):
        del store['not_a_real_id']


def test_num_docs(ten_simple_docs, qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index(ten_simple_docs)

    assert store.num_docs() == 10

    del store[ten_simple_docs[0].id]
    assert store.num_docs() == 9

    del store[ten_simple_docs[3].id, ten_simple_docs[5].id]
    assert store.num_docs() == 7

    more_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(5)]
    store.index(more_docs)
    assert store.num_docs() == 12

    del store[more_docs[2].id, ten_simple_docs[7].id]
    assert store.num_docs() == 10


def test_multimodal_doc(qdrant_config, qdrant):
    class MyMultiModalDoc(BaseDoc):
        image: ImageDoc
        text: TextDoc

    store = QdrantDocumentIndex[MyMultiModalDoc](db_config=qdrant_config)

    doc = [
        MyMultiModalDoc(
            image=ImageDoc(embedding=np.random.randn(128)), text=TextDoc(text='hello')
        )
    ]
    store.index(doc)

    id_ = doc[0].id
    assert store[id_].id == id_
    assert cosine(store[id_].image.embedding, doc[0].image.embedding) == pytest.approx(0.0)
    assert store[id_].text.text == doc[0].text.text
