from typing import Optional, Union

import numpy as np
import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.array import DocumentArrayStacked
from docarray.typing import NdArray, TorchTensor


@pytest.fixture()
def da():
    class Text(BaseDocument):
        text: str

    return DocumentArray([Text(text='hello') for _ in range(10)])


def test_iterate(da):
    for doc, doc2 in zip(da, da._data):
        assert doc.id == doc2.id


def test_append():
    class Text(BaseDocument):
        text: str

    da = DocumentArray[Text]([])

    da.append(Text(text='hello', id='1'))

    assert len(da) == 1
    assert da[0].id == '1'


def test_extend():
    class Text(BaseDocument):
        text: str

    da = DocumentArray[Text]([Text(text='hello', id=str(i)) for i in range(10)])

    da.extend([Text(text='hello', id=str(10 + i)) for i in range(10)])

    assert len(da) == 20
    for da, i in zip(da, range(20)):
        assert da.id == str(i)


def test_slice(da):
    da2 = da[0:5]
    assert type(da2) == da.__class__
    assert len(da2) == 5


def test_document_array():
    class Text(BaseDocument):
        text: str

    da = DocumentArray([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_document_array_fixed_type():
    class Text(BaseDocument):
        text: str

    da = DocumentArray[Text]([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_get_bulk_attributes_function():
    class Mmdoc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da._get_array_attribute('tensor')

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da._get_array_attribute('text')

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_set_attributes():
    class InnerDoc(BaseDocument):
        text: str

    class Mmdoc(BaseDocument):
        inner: InnerDoc

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N))
    )

    list_docs = [InnerDoc(text=f'hello{i}') for i in range(N)]
    da._set_array_attribute('inner', list_docs)

    for doc, list_doc in zip(da, list_docs):
        assert doc.inner is list_doc


def test_get_bulk_attributes():
    class Mmdoc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da.tensor

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da.text

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_get_bulk_attributes_document():
    class InnerDoc(BaseDocument):
        text: str

    class Mmdoc(BaseDocument):
        inner: InnerDoc

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N))
    )

    assert isinstance(da.inner, DocumentArray)


def test_get_bulk_attributes_optional_type():
    class Mmdoc(BaseDocument):
        text: str
        tensor: Optional[NdArray]

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da.tensor

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da.text

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_get_bulk_attributes_union_type():
    class Mmdoc(BaseDocument):
        text: str
        tensor: Union[NdArray, TorchTensor]

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da.tensor

    assert len(tensors) == N
    assert isinstance(tensors, list)
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da.text

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_get_bulk_attributes_union_type_nested():
    class MyDoc(BaseDocument):
        embedding: Union[Optional[TorchTensor], Optional[NdArray]]
        embedding2: Optional[Union[TorchTensor, NdArray]]
        embedding3: Optional[Optional[TorchTensor]]
        embedding4: Union[Optional[Union[TorchTensor, NdArray]], TorchTensor]

    da = DocumentArray[MyDoc](
        [
            MyDoc(
                embedding=torch.rand(10),
                embedding2=torch.rand(10),
                embedding3=torch.rand(10),
                embedding4=torch.rand(10),
            )
            for _ in range(10)
        ]
    )

    for attr in ['embedding', 'embedding2', 'embedding3', 'embedding4']:
        tensors = getattr(da, attr)
        assert len(tensors) == 10
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert tensor.shape == (10,)


def test_get_from_slice():
    class Doc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Doc](
        (Doc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocumentArray)

    tensors = da_sliced.tensor
    assert len(tensors) == 5
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da_sliced.text
    assert len(texts) == 5
    for i, text in enumerate(texts):
        assert text == f'hello{i*2}'


def test_get_from_slice_stacked():
    class Doc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Doc](
        (Doc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    ).stack()

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocumentArrayStacked)

    tensors = da_sliced.tensor
    assert tensors.shape == (5, 3, 224, 224)

    texts = da_sliced.text
    assert len(texts) == 5
    for i, text in enumerate(texts):
        assert text == f'hello{i * 2}'
