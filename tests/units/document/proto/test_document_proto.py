from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest
import torch

from docarray import DocList
from docarray.base_doc import AnyDoc, BaseDoc
from docarray.typing import NdArray, TorchTensor
from docarray.utils._internal.misc import is_tf_available

if is_tf_available():
    import tensorflow as tf


@pytest.mark.proto
def test_proto_simple():
    class CustomDoc(BaseDoc):
        text: str

    doc = CustomDoc(text='hello')

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_ndarray():
    class CustomDoc(BaseDoc):
        tensor: NdArray

    tensor = np.zeros((3, 224, 224))
    doc = CustomDoc(tensor=tensor)

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    assert (new_doc.tensor == tensor).all()


@pytest.mark.proto
def test_proto_with_nested_doc():
    class CustomInnerDoc(BaseDoc):
        tensor: NdArray

    class CustomDoc(BaseDoc):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(text='hello', inner=CustomInnerDoc(tensor=np.zeros((3, 224, 224))))

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_with_chunks_doc():
    class CustomInnerDoc(BaseDoc):
        tensor: NdArray

    class CustomDoc(BaseDoc):
        text: str
        chunks: DocList[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocList[CustomInnerDoc](
            [CustomInnerDoc(tensor=np.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):
        assert (chunk1.tensor == chunk2.tensor).all()


@pytest.mark.proto
def test_proto_with_nested_doc_pytorch():
    class CustomInnerDoc(BaseDoc):
        tensor: TorchTensor

    class CustomDoc(BaseDoc):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(
        text='hello', inner=CustomInnerDoc(tensor=torch.zeros((3, 224, 224)))
    )

    CustomDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_proto_with_chunks_doc_pytorch():
    class CustomInnerDoc(BaseDoc):
        tensor: TorchTensor

    class CustomDoc(BaseDoc):
        text: str
        chunks: DocList[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocList[CustomInnerDoc](
            [CustomInnerDoc(tensor=torch.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):
        assert (chunk1.tensor == chunk2.tensor).all()


@pytest.mark.proto
def test_optional_field_in_doc():
    class CustomDoc(BaseDoc):
        text: Optional[str] = None

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())


@pytest.mark.proto
def test_optional_field_nested_in_doc():
    class InnerDoc(BaseDoc):
        title: str

    class CustomDoc(BaseDoc):
        text: Optional[InnerDoc] = None

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())


@pytest.mark.proto
def test_integer_field():
    class Meow(BaseDoc):
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
    class MyDoc(BaseDoc):
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
    class MyDoc(BaseDoc):
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
    class MyDoc(BaseDoc):
        tensor: TorchTensor

    doc = MyDoc(tensor=torch.zeros([5, 5], dtype=dtype))
    assert doc.tensor.dtype == dtype
    assert MyDoc.from_protobuf(doc.to_protobuf()).tensor.dtype == dtype
    assert MyDoc.parse_obj(doc.dict()).tensor.dtype == dtype


@pytest.mark.proto
def test_nested_dict():
    class MyDoc(BaseDoc):
        data: Dict

    doc = MyDoc(data={'data': (1, 2)})

    MyDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_nested_dict_error():
    class MyDoc(BaseDoc):
        data: Dict

    doc = MyDoc(data={0: (1, 2)})

    with pytest.raises(ValueError, match="Protobuf only support string as key"):
        doc.to_protobuf()


@pytest.mark.proto
def test_tuple_complex():
    class MyDoc(BaseDoc):
        data: Tuple

    doc = MyDoc(data=(1, 2))

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc2.data == (1, 2)


@pytest.mark.proto
def test_list_complex():
    class MyDoc(BaseDoc):
        data: List

    doc = MyDoc(data=[(1, 2)])

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc2.data == [(1, 2)]


@pytest.mark.proto
def test_nested_tensor_list():
    class MyDoc(BaseDoc):
        data: List

    doc = MyDoc(data=[np.zeros(10)])

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert isinstance(doc2.data[0], np.ndarray)
    assert isinstance(doc2.data[0], NdArray)

    assert (doc2.data[0] == np.zeros(10)).all()


@pytest.mark.proto
def test_nested_tensor_dict():
    class MyDoc(BaseDoc):
        data: Dict

    doc = MyDoc(data={'hello': np.zeros(10)})

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    assert isinstance(doc2.data['hello'], np.ndarray)
    assert isinstance(doc2.data['hello'], NdArray)

    assert (doc2.data['hello'] == np.zeros(10)).all()


@pytest.mark.proto
def test_super_complex_nested():
    class MyDoc(BaseDoc):
        data: Dict

    data = {'hello': (torch.zeros(55), 1, 'hi', [torch.ones(55), np.zeros(10), (1, 2)])}
    doc = MyDoc(data=data)

    doc2 = MyDoc.from_protobuf(doc.to_protobuf())

    (doc2.data['hello'][3][0] == torch.ones(55)).all()


@pytest.mark.tensorflow
def test_super_complex_nested_tensorflow():
    class MyDoc(BaseDoc):
        data: Dict

    data = {'hello': (torch.zeros(55), 1, 'hi', [tf.ones(55), np.zeros(10), (1, 2)])}
    doc = MyDoc(data=data)

    MyDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_any_doc_proto():
    doc = AnyDoc(hello='world')
    pt = doc.to_protobuf()
    doc2 = AnyDoc.from_protobuf(pt)
    assert doc2.hello == 'world'


@pytest.mark.proto
def test_nested_list():
    from typing import List

    from docarray import BaseDoc, DocList
    from docarray.documents import TextDoc

    class TextDocWithId(TextDoc):
        id: str

    class ResultTestDoc(BaseDoc):
        matches: List[TextDocWithId]

    da = DocList[ResultTestDoc](
        [
            ResultTestDoc(matches=[TextDocWithId(id=f'{i}') for _ in range(10)])
            for i in range(10)
        ]
    )

    DocList[ResultTestDoc].from_protobuf(da.to_protobuf())


@pytest.mark.proto
def test_nested_dict_typed():
    from docarray import BaseDoc, DocList
    from docarray.documents import TextDoc

    class TextDocWithId(TextDoc):
        id: str

    class ResultTestDoc(BaseDoc):
        matches: Dict[str, TextDocWithId]

    da = DocList[ResultTestDoc](
        [
            ResultTestDoc(matches={f'{i}': TextDocWithId(id=f'{i}') for _ in range(10)})
            for i in range(10)
        ]
    )

    DocList[ResultTestDoc].from_protobuf(da.to_protobuf())
