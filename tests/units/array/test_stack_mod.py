from typing import Union

import numpy as np
import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.typing import AnyTensor, NdArray, TorchTensor


@pytest.fixture()
def batch():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    return batch.stack()


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
        tensor: Union[TorchTensor[3, 224, 224], NdArray[3, 224, 224]]

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
