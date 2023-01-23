from typing import Dict, Optional, Union

import numpy as np
import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.array import DocumentArrayStacked
from docarray.documents import Image
from docarray.typing import AnyEmbedding, AnyTensor, NdArray, TorchTensor


@pytest.fixture()
def batch():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    return batch.stack()


def test_repr(batch):
    assert batch.__repr__() == '<DocumentArrayStacked[Image] (length=10)>'


def test_len(batch):
    assert len(batch) == 10


def test_getitem(batch):
    for i in range(len(batch)):
        assert (batch[i].tensor == torch.zeros(3, 224, 224)).all()


def test_iterator(batch):
    for doc in batch:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_stack_setter(batch):

    batch.tensor = torch.ones(10, 3, 224, 224)

    assert (batch.tensor == torch.ones(10, 3, 224, 224)).all()


def test_stack_optional(batch):

    assert (batch._columns['tensor'] == torch.zeros(10, 3, 224, 224)).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()


def test_stack_numpy():
    class Image(BaseDocument):
        tensor: NdArray[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    batch = batch.stack()

    assert (batch._columns['tensor'] == np.zeros((10, 3, 224, 224))).all()
    assert (batch.tensor == np.zeros((10, 3, 224, 224))).all()
    assert batch.tensor.ctypes.data == batch._columns['tensor'].ctypes.data

    batch.unstack()


def test_stack(batch):

    assert (batch._columns['tensor'] == torch.zeros(10, 3, 224, 224)).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()
    assert batch._columns['tensor'].data_ptr() == batch.tensor.data_ptr()

    for doc, tensor in zip(batch, batch.tensor):
        assert doc.tensor.data_ptr() == tensor.data_ptr()

    for i in range(len(batch)):
        assert batch[i].tensor.data_ptr() == batch.tensor[i].data_ptr()


def test_stack_mod_nested_document():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(BaseDocument):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = batch.stack()

    assert (
        batch._columns['img']._columns['tensor'] == torch.zeros(10, 3, 224, 224)
    ).all()

    assert (batch.img.tensor == torch.zeros(10, 3, 224, 224)).all()

    assert (
        batch._columns['img']._columns['tensor'].data_ptr()
        == batch.img.tensor.data_ptr()
    )


def test_convert_to_da(batch):
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch = batch.stack()
    da = batch.unstack()

    for doc in da:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_unstack_nested_document():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(BaseDocument):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = batch.stack()

    da = batch.unstack()

    for doc in da:
        assert (doc.img.tensor == torch.zeros(3, 224, 224)).all()


def test_proto_stacked_mode_torch(batch):

    batch.from_protobuf(batch.to_protobuf())


def test_proto_stacked_mode_numpy():
    class MyDoc(BaseDocument):
        tensor: NdArray[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    da = da.stack()

    da.from_protobuf(da.to_protobuf())


def test_stack_call():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    da = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    da = da.stack()

    assert len(da) == 10

    assert da.tensor.shape == (10, 3, 224, 224)


def test_context_manager():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    da = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    with da.stacked_mode() as da:
        assert len(da) == 10

        assert da.tensor.shape == (10, 3, 224, 224)

        da.tensor = torch.ones(10, 3, 224, 224)

    tensor = da.tensor

    assert isinstance(tensor, list)
    for doc in da:
        assert (doc.tensor == torch.ones(3, 224, 224)).all()


def test_stack_union():
    class Image(BaseDocument):
        tensor: Union[NdArray[3, 224, 224], TorchTensor[3, 224, 224]]

    batch = DocumentArray[Image](
        [Image(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )
    batch[3].tensor = np.zeros((3, 224, 224))

    # union fields aren't actually stacked
    # just checking that there is no error
    batch.stack()


@pytest.mark.parametrize(
    'tensor_type,tensor',
    [(TorchTensor, torch.zeros(3, 224, 224)), (NdArray, np.zeros((3, 224, 224)))],
)
def test_any_tensor_with_torch(tensor_type, tensor):
    class Image(BaseDocument):
        tensor: AnyTensor

    da = DocumentArray[Image](
        [Image(tensor=tensor) for _ in range(10)],
        tensor_type=tensor_type,
    ).stack()

    for i in range(len(da)):
        assert (da[i].tensor == tensor).all()

    assert 'tensor' in da._columns.keys()
    assert isinstance(da._columns['tensor'], tensor_type)


def test_any_tensor_with_optional():
    tensor = torch.zeros(3, 224, 224)

    class Image(BaseDocument):
        tensor: Optional[AnyTensor]

    class TopDoc(BaseDocument):
        img: Image

    da = DocumentArray[TopDoc](
        [TopDoc(img=Image(tensor=tensor)) for _ in range(10)],
        tensor_type=TorchTensor,
    ).stack()

    for i in range(len(da)):
        assert (da.img[i].tensor == tensor).all()

    assert 'tensor' in da.img._columns.keys()
    assert isinstance(da.img._columns['tensor'], TorchTensor)


def test_dict_stack():
    class MyDoc(BaseDocument):
        my_dict: Dict[str, int]

    da = DocumentArray[MyDoc](
        [MyDoc(my_dict={'a': 1, 'b': 2}) for _ in range(10)]
    ).stack()

    da.my_dict


def test_get_from_slice_stacked():
    class Doc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Doc](
        [Doc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N)]
    ).stack()

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocumentArrayStacked)

    tensors = da_sliced.tensor
    assert tensors.shape == (5, 3, 224, 224)

    texts = da_sliced.text
    assert len(texts) == 5
    for i, text in enumerate(texts):
        assert text == f'hello{i * 2}'


def test_stack_embedding():
    class MyDoc(BaseDocument):
        embedding: AnyEmbedding

    da = DocumentArray[MyDoc](
        [MyDoc(embedding=np.zeros(10)) for _ in range(10)]
    ).stack()

    assert 'embedding' in da._columns.keys()
    assert (da.embedding == np.zeros((10, 10))).all()


@pytest.mark.parametrize('tensor_backend', [TorchTensor, NdArray])
def test_stack_none(tensor_backend):
    class MyDoc(BaseDocument):
        tensor: Optional[AnyTensor]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=None) for _ in range(10)], tensor_type=tensor_backend
    ).stack()

    assert 'tensor' in da._columns.keys()


def test_to_device():
    da = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 5))], tensor_type=TorchTensor
    )
    da = da.stack()
    assert da.tensor.device == torch.device('cpu')
    da.to('meta')
    assert da.tensor.device == torch.device('meta')


def test_to_device_nested():
    class MyDoc(BaseDocument):
        tensor: TorchTensor
        docs: Image

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=torch.zeros(3, 5), docs=Image(tensor=torch.zeros(3, 5)))],
        tensor_type=TorchTensor,
    )
    da = da.stack()
    assert da.tensor.device == torch.device('cpu')
    assert da.docs.tensor.device == torch.device('cpu')
    da.to('meta')
    assert da.tensor.device == torch.device('meta')
    assert da.docs.tensor.device == torch.device('meta')


def test_to_device_numpy():
    da = DocumentArray[Image]([Image(tensor=np.zeros((3, 5)))], tensor_type=NdArray)
    da = da.stack()
    with pytest.raises(NotImplementedError):
        da.to('meta')


def test_keep_dtype_torch():
    class MyDoc(BaseDocument):
        tensor: TorchTensor

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=torch.zeros([2, 4], dtype=torch.int32)) for _ in range(3)]
    )
    assert da[0].tensor.dtype == torch.int32

    da = da.stack()
    assert da[0].tensor.dtype == torch.int32
    assert da.tensor.dtype == torch.int32


def test_keep_dtype_np():
    class MyDoc(BaseDocument):
        tensor: NdArray

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros([2, 4], dtype=np.int32)) for _ in range(3)]
    )
    assert da[0].tensor.dtype == np.int32

    da = da.stack()
    assert da[0].tensor.dtype == np.int32
    assert da.tensor.dtype == np.int32
