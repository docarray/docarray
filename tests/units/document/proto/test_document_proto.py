from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest
import torch

from docarray import DocumentArray
from docarray.base_document import BaseDocument
from docarray.typing import NdArray, TorchTensor
from docarray.utils.misc import is_tf_available

if is_tf_available():
    import tensorflow as tf


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
        int,
        np.int8,
        np.int64,
        float,
        np.float16,
        np.longfloat,
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
def test_nested_dict():
    class MyDoc(BaseDocument):
        data: Dict

    doc = MyDoc(data={'data': (1, 2)})

    MyDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_tuple_complex():
    class MyDoc(BaseDocument):
        data: Tuple

    doc = MyDoc(data=(1, 2))

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc2.data == (1, 2)


@pytest.mark.proto
def test_list_complex():
    class MyDoc(BaseDocument):
        data: List

    doc = MyDoc(data=[(1, 2)])

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc2.data == [(1, 2)]


@pytest.mark.proto
def test_nested_tensor_list():
    class MyDoc(BaseDocument):
        data: List

    doc = MyDoc(data=[np.zeros(10)])

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert isinstance(doc2.data[0], np.ndarray)
    assert isinstance(doc2.data[0], NdArray)

    assert (doc2.data[0] == np.zeros(10)).all()


@pytest.mark.proto
def test_super_complex_nested():
    class MyDoc(BaseDocument):
        data: Dict

    data = {'hello': (torch.zeros(55), 1, 'hi', [torch.ones(55), np.zeros(10), (1, 2)])}
    doc = MyDoc(data=data)

    MyDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.tensorflow
def test_super_complex_nested_tensorflow():
    class MyDoc(BaseDocument):
        data: Dict

    data = {'hello': (torch.zeros(55), 1, 'hi', [tf.ones(55), np.zeros(10), (1, 2)])}
    doc = MyDoc(data=data)

    MyDoc.from_protobuf(doc.to_protobuf())
