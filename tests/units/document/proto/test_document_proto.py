from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest
import torch

from docarray import DocumentArray
from docarray.base_document import BaseDocument
from docarray.typing import AnyUrl, NdArray, TorchTensor


@pytest.mark.proto
def test_proto_simple():
    class CustomDoc(BaseDocument):
        text: str

    doc = CustomDoc(text='hello')

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_ndarray():
    class CustomDoc(BaseDocument):
        tensor: NdArray

    tensor = np.zeros((3, 224, 224))
    doc = CustomDoc(tensor=tensor)

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    assert (new_doc.tensor == tensor).all()


@pytest.mark.proto
def test_proto_with_nested_doc():
    class CustomInnerDoc(BaseDocument):
        tensor: NdArray

    class CustomDoc(BaseDocument):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(text='hello', inner=CustomInnerDoc(tensor=np.zeros((3, 224, 224))))

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_with_chunks_doc():
    class CustomInnerDoc(BaseDocument):
        tensor: NdArray

    class CustomDoc(BaseDocument):
        text: str
        chunks: DocumentArray[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocumentArray[CustomInnerDoc](
            [CustomInnerDoc(tensor=np.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):
        assert (chunk1.tensor == chunk2.tensor).all()


@pytest.mark.proto
def test_proto_with_nested_doc_pytorch():
    class CustomInnerDoc(BaseDocument):
        tensor: TorchTensor

    class CustomDoc(BaseDocument):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(
        text='hello', inner=CustomInnerDoc(tensor=torch.zeros((3, 224, 224)))
    )

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_with_chunks_doc_pytorch():
    class CustomInnerDoc(BaseDocument):
        tensor: TorchTensor

    class CustomDoc(BaseDocument):
        text: str
        chunks: DocumentArray[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocumentArray[CustomInnerDoc](
            [CustomInnerDoc(tensor=torch.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):
        assert (chunk1.tensor == chunk2.tensor).all()


@pytest.mark.proto
def test_optional_field_in_doc():
    class CustomDoc(BaseDocument):
        text: Optional[str]

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())


@pytest.mark.proto
def test_optional_field_nested_in_doc():
    class InnerDoc(BaseDocument):
        title: str

    class CustomDoc(BaseDocument):
        text: Optional[InnerDoc]

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())


@pytest.mark.proto
def test_integer_field():
    class Meow(BaseDocument):
        age: int
        wealth: float
        registered: bool

    d = Meow(age=30, wealth=100.5, registered=True)
    rebuilt_doc = Meow.from_protobuf(d.to_protobuf())
    assert rebuilt_doc.age == 30
    assert rebuilt_doc.wealth == 100.5
    assert rebuilt_doc.registered


@pytest.mark.proto
def test_list_set_dict_tuple_field():
    class MyDoc(BaseDocument):
        list_: List
        dict_: Dict
        tuple_: Tuple
        set_: Set

    d = MyDoc(
        list_=[0, 1, 2], dict_={'a': 0, 'b': 1}, tuple_=tuple([0, 1]), set_={0, 1}
    )
    rebuilt_doc = MyDoc.from_protobuf(d.to_protobuf())
    assert rebuilt_doc.list_ == [0, 1, 2]
    assert rebuilt_doc.dict_ == {'a': 0, 'b': 1}
    assert rebuilt_doc.tuple_ == (0, 1)
    assert rebuilt_doc.set_ == {0, 1}


@pytest.mark.proto
@pytest.mark.parametrize(
    'dtype',
    [
        np.uint,
        np.uint8,
        np.uint64,
        np.int,
        np.int8,
        np.int64,
        np.float,
        np.float16,
        np.float128,
        np.double,
    ],
)
def test_ndarray_dtype(dtype):
    class MyDoc(BaseDocument):
        tensor: NdArray

    doc = MyDoc(tensor=np.ndarray([1, 2, 3], dtype=dtype))
    assert doc.tensor.dtype == dtype
    assert MyDoc.from_protobuf(doc.to_protobuf()).tensor.dtype == dtype
    assert MyDoc.parse_obj(doc.dict()).tensor.dtype == dtype


@pytest.mark.proto
@pytest.mark.parametrize(
    'dtype',
    [
        torch.uint8,
        torch.int,
        torch.int8,
        torch.int64,
        torch.float,
        torch.float64,
        torch.double,
    ],
)
def test_torch_dtype(dtype):
    class MyDoc(BaseDocument):
        tensor: TorchTensor

    doc = MyDoc(tensor=torch.zeros([5, 5], dtype=dtype))
    assert doc.tensor.dtype == dtype
    assert MyDoc.from_protobuf(doc.to_protobuf()).tensor.dtype == dtype
    assert MyDoc.parse_obj(doc.dict()).tensor.dtype == dtype


@pytest.mark.proto
def test_proto_schema_map():
    class A(BaseDocument):
        url: AnyUrl
        tensor: TorchTensor

    class B(BaseDocument):
        link: AnyUrl
        array: TorchTensor

    a = A(url='hello', tensor=torch.zeros(3))

    with pytest.raises(ValueError):
        B.from_protobuf(a.to_protobuf())

    b = B.from_protobuf_field_map(
        a.to_protobuf(), field_map={'url': 'link', 'tensor': 'array'}
    )

    assert b.link == a.url
    assert (b.array == a.tensor).all()
