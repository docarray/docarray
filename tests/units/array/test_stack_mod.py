import pytest
import torch

from docarray import Document, DocumentArray
from docarray.typing import TorchTensor


def test_stack():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch.stacked()

    assert (batch._columns['tensor'] == torch.zeros(10, 3, 224, 224)).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()
    assert batch._columns['tensor'].data_ptr() == batch.tensor.data_ptr()

    for doc, tensor in zip(batch, batch.tensor):
        assert doc.tensor.data_ptr() == tensor.data_ptr()


def test_stack_mod_nested_document():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(Document):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch.stacked()

    assert (
        batch._columns['img']._columns['tensor'] == torch.zeros(10, 3, 224, 224)
    ).all()

    assert (batch.img.tensor == torch.zeros(10, 3, 224, 224)).all()

    assert (
        batch._columns['img']._columns['tensor'].data_ptr()
        == batch.img.tensor.data_ptr()
    )


def test_unstack():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch.stacked()

    batch.unstacked()

    for doc in batch:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_unstack_nested_document():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(Document):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch.stacked()

    batch.unstacked()


def test_stack_runtime_error():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch.stacked()

    with pytest.raises(RuntimeError):
        batch.append([])
