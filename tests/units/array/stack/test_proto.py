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
def test_proto_none_column():
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
