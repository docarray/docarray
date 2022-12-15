import numpy as np
import pytest
import torch

from docarray import Document, DocumentArray
from docarray.array.array_stacked import DocumentArrayStacked
from docarray.typing import NdArray, TorchTensor


@pytest.fixture()
def batch():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    return DocumentArrayStacked[Image](batch)


def test_len(batch):
    assert len(batch) == 10


def test_getitem(batch):
    for i in range(len(batch)):
        print(i)
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
    class Image(Document):
        tensor: NdArray[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    batch = DocumentArrayStacked[Image](batch)

    assert (batch._columns['tensor'] == np.zeros((10, 3, 224, 224))).all()
    assert (batch.tensor == np.zeros((10, 3, 224, 224))).all()
    assert batch.tensor.ctypes.data == batch._columns['tensor'].ctypes.data

    batch.to_document_array(batch)


def test_stack(batch):

    assert (batch._columns['tensor'] == torch.zeros(10, 3, 224, 224)).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()
    assert batch._columns['tensor'].data_ptr() == batch.tensor.data_ptr()

    for doc, tensor in zip(batch, batch.tensor):
        assert doc.tensor.data_ptr() == tensor.data_ptr()

    for i in range(len(batch)):
        assert batch[i].tensor.data_ptr() == batch.tensor[i].data_ptr()


def test_stack_mod_nested_document():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(Document):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = DocumentArrayStacked[MMdoc](batch)

    assert (
        batch._columns['img']._columns['tensor'] == torch.zeros(10, 3, 224, 224)
    ).all()

    assert (batch.img.tensor == torch.zeros(10, 3, 224, 224)).all()

    assert (
        batch._columns['img']._columns['tensor'].data_ptr()
        == batch.img.tensor.data_ptr()
    )


def test_convert_to_da(batch):
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch = DocumentArrayStacked[Image](batch)
    da = batch.to_document_array(batch)

    for doc in da:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_unstack_nested_document():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(Document):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = DocumentArrayStacked[MMdoc](batch)

    da = batch.to_document_array(batch)

    for doc in da:
        assert (doc.img.tensor == torch.zeros(3, 224, 224)).all()


def test_proto_stacked_mode_torch(batch):

    batch.from_protobuf(batch.to_protobuf())


def test_proto_stacked_mode_numpy():
    class MyDoc(Document):
        tensor: NdArray[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    da = DocumentArrayStacked[MyDoc](da)

    da.from_protobuf(da.to_protobuf())


def test_stack_call():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    da = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    da = da.stack()

    assert len(da) == 10

    assert da.tensor.shape == (10, 3, 224, 224)
