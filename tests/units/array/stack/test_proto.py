from typing import Dict, Union

import numpy as np
import pytest
import torch

from docarray import BaseDoc, DocList
from docarray.array import DocVec
from docarray.typing import NdArray, TorchTensor


@pytest.fixture()
def batch():
    class Image(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocList[Image]([Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)])

    return batch.to_doc_vec()


@pytest.mark.proto
def test_proto_stacked_mode_torch(batch):
    batch.from_protobuf(batch.to_protobuf())


@pytest.mark.proto
def test_proto_stacked_mode_numpy():
    class MyDoc(BaseDoc):
        tensor: NdArray[3, 224, 224]

    da = DocList[MyDoc]([MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)])

    da = da.to_doc_vec()

    da.from_protobuf(da.to_protobuf())


@pytest.mark.proto
def test_stacked_proto():
    class CustomDocument(BaseDoc):
        image: NdArray

    da = DocList[CustomDocument](
        [CustomDocument(image=np.zeros((3, 224, 224))) for _ in range(10)]
    ).to_doc_vec()

    da2 = DocVec[CustomDocument].from_protobuf(da.to_protobuf())

    assert isinstance(da2, DocVec)
    assert da.doc_type == da2.doc_type
    assert (da2.image == da.image).all()


@pytest.mark.proto
def test_proto_none_tensor_column():
    class MyOtherDoc(BaseDoc):
        embedding: Union[NdArray, None]
        other_embedding: NdArray
        third_embedding: Union[NdArray, None]

    da = DocVec[MyOtherDoc](
        [
            MyOtherDoc(
                other_embedding=np.random.random(512),
            ),
            MyOtherDoc(other_embedding=np.random.random(512)),
        ]
    )
    assert da._storage.tensor_columns['embedding'] is None
    assert da._storage.tensor_columns['other_embedding'] is not None
    assert da._storage.tensor_columns['third_embedding'] is None

    proto = da.to_protobuf()
    da_after = DocVec[MyOtherDoc].from_protobuf(proto)

    assert da_after._storage.tensor_columns['embedding'] is None
    assert da_after._storage.tensor_columns['other_embedding'] is not None
    assert (
        da_after._storage.tensor_columns['other_embedding']
        == da._storage.tensor_columns['other_embedding']
    ).all()
    assert da_after._storage.tensor_columns['third_embedding'] is None


@pytest.mark.proto
def test_proto_none_doc_column():
    class InnerDoc(BaseDoc):
        embedding: NdArray

    class MyDoc(BaseDoc):
        inner: Union[InnerDoc, None]
        other_inner: Union[InnerDoc, None]

    da = DocVec[MyDoc](
        [
            MyDoc(other_inner=InnerDoc(embedding=np.random.random(512))),
            MyDoc(other_inner=InnerDoc(embedding=np.random.random(512))),
        ]
    )
    assert da._storage.doc_columns['inner'] is None
    assert len(da._storage.doc_columns['other_inner']) == 2

    proto = da.to_protobuf()
    da_after = DocVec[MyDoc].from_protobuf(proto)

    assert da_after._storage.doc_columns['inner'] is None
    assert len(da._storage.doc_columns['other_inner']) == 2
    assert (da.other_inner.embedding == da_after.other_inner.embedding).all()


@pytest.mark.proto
def test_proto_none_docvec_column():
    class InnerDoc(BaseDoc):
        embedding: NdArray

    class MyDoc(BaseDoc):
        inner_l: Union[DocList[InnerDoc], None]
        inner_v: Union[DocVec[InnerDoc], None]
        inner_exists_v: Union[DocVec[InnerDoc], None]
        inner_exists_l: Union[DocList[InnerDoc], None]

    def _make_inner_list():
        return DocList[InnerDoc](
            [
                InnerDoc(embedding=np.random.random(512)),
                InnerDoc(embedding=np.random.random(512)),
            ]
        )

    da = DocVec[MyDoc](
        [
            MyDoc(
                inner_exists_l=_make_inner_list(),
                inner_exists_v=_make_inner_list().to_doc_vec(),
            ),
            MyDoc(
                inner_exists_l=_make_inner_list(),
                inner_exists_v=_make_inner_list().to_doc_vec(),
            ),
        ]
    )
    assert da._storage.docs_vec_columns['inner_l'] is None
    assert da._storage.docs_vec_columns['inner_v'] is None
    assert len(da._storage.docs_vec_columns['inner_exists_l']) == 2
    assert len(da._storage.docs_vec_columns['inner_exists_v']) == 2
    assert da.inner_exists_l[0].embedding.shape == (2, 512)
    assert da.inner_exists_l[1].embedding.shape == (2, 512)
    assert da.inner_exists_v[0].embedding.shape == (2, 512)
    assert da.inner_exists_v[1].embedding.shape == (2, 512)

    proto = da.to_protobuf()
    da_after = DocVec[MyDoc].from_protobuf(proto)

    assert da_after._storage.docs_vec_columns['inner_l'] is None
    assert da_after._storage.docs_vec_columns['inner_v'] is None
    assert len(da._storage.docs_vec_columns['inner_exists_l']) == 2
    assert len(da._storage.docs_vec_columns['inner_exists_v']) == 2
    assert (
        da.inner_exists_l[0].embedding == da_after.inner_exists_l[0].embedding
    ).all()
    assert (
        da.inner_exists_l[1].embedding == da_after.inner_exists_l[1].embedding
    ).all()
    assert (
        da.inner_exists_v[0].embedding == da_after.inner_exists_v[0].embedding
    ).all()
    assert (
        da.inner_exists_v[1].embedding == da_after.inner_exists_v[1].embedding
    ).all()


@pytest.mark.proto
def test_proto_any_column():
    class MyDoc(BaseDoc):
        embedding: NdArray
        text: str
        d: Dict

    da = DocVec[MyDoc](
        [
            MyDoc(
                embedding=np.random.random(512),
                text='hi',
                d={'a': 1},
            ),
            MyDoc(embedding=np.random.random(512), text='there', d={'b': 2}),
        ]
    )
    assert da._storage.tensor_columns['embedding'].shape == (2, 512)
    assert da._storage.any_columns['text'] == ['hi', 'there']
    assert da._storage.any_columns['d'] == [{'a': 1}, {'b': 2}]

    proto = da.to_protobuf()
    da_after = DocVec[MyDoc].from_protobuf(proto)

    assert da_after.doc_type == da.doc_type
    assert da._storage.tensor_columns['embedding'].shape == (2, 512)
    assert (
        da_after._storage.tensor_columns['embedding']
        == da._storage.tensor_columns['embedding']
    ).all()
    assert da._storage.any_columns['text'] == ['hi', 'there']
    assert da._storage.any_columns['d'] == [{'a': 1}, {'b': 2}]

    assert (da_after.embedding == da.embedding).all()
    assert da_after.text == da.text
    assert da_after.d == da.d
