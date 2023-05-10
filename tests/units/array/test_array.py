from typing import Optional, TypeVar, Union

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl, NdArray, TorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing import TensorFlowTensor


@pytest.fixture()
def da():
    class Text(BaseDoc):
        text: str

    return DocList[Text]([Text(text=f'hello {i}') for i in range(10)])


def test_iterate(da):
    for doc, doc2 in zip(da, da):
        assert doc.id == doc2.id


def test_append():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([])

    da.append(Text(text='hello', id='1'))

    assert len(da) == 1
    assert da[0].id == '1'


def test_extend():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([Text(text='hello', id=str(i)) for i in range(10)])

    da.extend([Text(text='hello', id=str(10 + i)) for i in range(10)])

    assert len(da) == 20
    for da, i in zip(da, range(20)):
        assert da.id == str(i)


def test_extend_itself():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([Text(text='hello', id=str(i)) for i in range(10)])

    da.extend(da)

    assert len(da) == 20


def test_slice(da):
    da2 = da[0:5]
    assert type(da2) == da.__class__
    assert len(da2) == 5


def test_document_array():
    class Text(BaseDoc):
        text: str

    da = DocList([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_empty_array():
    da = DocList()
    len(da) == 0


def test_document_array_fixed_type():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_ndarray_equality():
    class Text(BaseDoc):
        tensor: NdArray

    arr1 = Text(tensor=np.zeros(5))
    arr2 = Text(tensor=np.zeros(5))
    arr3 = Text(tensor=np.ones(5))
    arr4 = Text(tensor=np.zeros(4))

    assert arr1 == arr2
    assert arr1 != arr3
    assert arr1 != arr4


def test_tensor_equality():
    class Text(BaseDoc):
        tensor: TorchTensor

    torch1 = Text(tensor=torch.zeros(128))
    torch2 = Text(tensor=torch.zeros(128))
    torch3 = Text(tensor=torch.zeros(126))
    torch4 = Text(tensor=torch.ones(128))

    assert torch1 == torch2
    assert torch1 != torch3
    assert torch1 != torch4


def test_documentarray():
    class Text(BaseDoc):
        text: str

    da1 = DocList([Text(text='hello')])
    da2 = DocList([Text(text='hello')])

    assert da1 == da2
    assert da1 == [Text(text='hello') for _ in range(len(da1))]
    assert da2 == [Text(text='hello') for _ in range(len(da2))]


@pytest.mark.tensorflow
def test_tensorflowtensor_equality():
    class Text(BaseDoc):
        tensor: TensorFlowTensor

    tensor1 = Text(tensor=tf.constant([1, 2, 3, 4, 5, 6]))
    tensor2 = Text(tensor=tf.constant([1, 2, 3, 4, 5, 6]))
    tensor3 = Text(tensor=tf.constant([[1.0, 2.0], [3.0, 5.0]]))
    tensor4 = Text(tensor=tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    assert tensor1 == tensor2
    assert tensor1 != tensor3
    assert tensor1 != tensor4


def test_text_tensor():
    class Text1(BaseDoc):
        tensor: NdArray

    class Text2(BaseDoc):
        tensor: TorchTensor

    arr_tensor1 = Text1(tensor=np.zeros(2))
    arr_tensor2 = Text2(tensor=torch.zeros(2))

    assert arr_tensor1 == arr_tensor2


def test_get_bulk_attributes_function():
    class Mmdoc(BaseDoc):
        text: str
        tensor: NdArray

    N = 10

    da = DocList[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da._get_data_column('tensor')

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da._get_data_column('text')

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_set_attributes():
    class InnerDoc(BaseDoc):
        text: str

    class Mmdoc(BaseDoc):
        inner: InnerDoc

    N = 10

    da = DocList[Mmdoc]((Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N)))

    list_docs = [InnerDoc(text=f'hello{i}') for i in range(N)]
    da._set_data_column('inner', list_docs)

    for doc, list_doc in zip(da, list_docs):
        assert doc.inner == list_doc


def test_get_bulk_attributes():
    class Mmdoc(BaseDoc):
        text: str
        tensor: NdArray

    N = 10

    da = DocList[Mmdoc](
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
    class InnerDoc(BaseDoc):
        text: str

    class Mmdoc(BaseDoc):
        inner: InnerDoc

    N = 10

    da = DocList[Mmdoc]((Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N)))

    assert isinstance(da.inner, DocList)


def test_get_bulk_attributes_optional_type():
    class Mmdoc(BaseDoc):
        text: str
        tensor: Optional[NdArray]

    N = 10

    da = DocList[Mmdoc](
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
    class Mmdoc(BaseDoc):
        text: str
        tensor: Union[NdArray, TorchTensor]

    N = 10

    da = DocList[Mmdoc](
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


@pytest.mark.tensorflow
def test_get_bulk_attributes_union_type_nested():
    class MyDoc(BaseDoc):
        embedding: Union[Optional[TorchTensor], Optional[NdArray]]
        embedding2: Optional[Union[TorchTensor, NdArray, TensorFlowTensor]]
        embedding3: Optional[Optional[TorchTensor]]
        embedding4: Union[
            Optional[Union[TorchTensor, NdArray, TensorFlowTensor]], TorchTensor
        ]

    da = DocList[MyDoc](
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
    class Doc(BaseDoc):
        text: str
        tensor: NdArray

    N = 10

    da = DocList[Doc](
        (Doc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocList)

    tensors = da_sliced.tensor
    assert len(tensors) == 5
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da_sliced.text
    assert len(texts) == 5
    for i, text in enumerate(texts):
        assert text == f'hello{i * 2}'


def test_del_item(da):
    assert len(da) == 10
    del da[2]
    assert len(da) == 9
    assert da.text == [
        'hello 0',
        'hello 1',
        'hello 3',
        'hello 4',
        'hello 5',
        'hello 6',
        'hello 7',
        'hello 8',
        'hello 9',
    ]
    del da[0:2]
    assert len(da) == 7
    assert da.text == [
        'hello 3',
        'hello 4',
        'hello 5',
        'hello 6',
        'hello 7',
        'hello 8',
        'hello 9',
    ]


def test_generic_type_var():
    T = TypeVar('T', bound=BaseDoc)

    def f(a: DocList[T]) -> DocList[T]:
        return a

    def g(a: DocList['BaseDoc']) -> DocList['BaseDoc']:
        return a

    a = DocList()
    f(a)
    g(a)


def test_construct():
    class Text(BaseDoc):
        text: str

    docs = [Text(text=f'hello {i}') for i in range(10)] + [BaseDoc()]

    da = DocList[Text].construct(docs)

    assert type(da[-1]) == BaseDoc


def test_reverse():
    class Text(BaseDoc):
        text: str

    docs = [Text(text=f'hello {i}') for i in range(10)]

    da = DocList[Text](docs)
    da.reverse()
    assert da[-1].text == 'hello 0'
    assert da[0].text == 'hello 9'


class Image(BaseDoc):
    tensor: Optional[NdArray]
    url: ImageUrl


def test_remove():
    images = [Image(url=f'http://url.com/foo_{i}.png') for i in range(3)]
    da = DocList[Image](images)
    da.remove(images[1])
    assert len(da) == 2
    assert da[0] == images[0]
    assert da[1] == images[2]


def test_pop():
    images = [Image(url=f'http://url.com/foo_{i}.png') for i in range(3)]
    da = DocList[Image](images)
    popped = da.pop(1)
    assert len(da) == 2
    assert popped == images[1]
    assert da[0] == images[0]
    assert da[1] == images[2]


def test_sort():
    images = [
        Image(url=f'http://url.com/foo_{i}.png', tensor=NdArray(i)) for i in [2, 0, 1]
    ]
    da = DocList[Image](images)
    da.sort(key=lambda img: len(img.tensor))
    assert len(da) == 3
    assert da[0].url == 'http://url.com/foo_0.png'
    assert da[1].url == 'http://url.com/foo_1.png'


def test_optional_field():
    from typing import Optional

    from docarray import BaseDoc, DocList
    from docarray.typing import ImageUrl, NdArray

    class Nested(BaseDoc):
        tensor: NdArray

    class Image(BaseDoc):
        url: ImageUrl
        features: Optional[Nested] = None

    docs = DocList[Image]([Image(url='http://url.com/foo.png') for _ in range(10)])

    assert docs.features == [None for _ in range(10)]
    assert isinstance(docs.features, list)
    assert not isinstance(docs.features, DocList)


def test_validate_list_dict():
    images = [
        dict(url=f'http://url.com/foo_{i}.png', tensor=NdArray(i)) for i in [2, 0, 1]
    ]

    docs = parse_obj_as(DocList[Image], images)

    assert docs.url == [
        'http://url.com/foo_2.png',
        'http://url.com/foo_0.png',
        'http://url.com/foo_1.png',
    ]
