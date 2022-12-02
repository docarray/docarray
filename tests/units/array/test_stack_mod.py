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

    batch.stack()

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

    batch.stack()

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

    batch.stack()

    batch.unstack()

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

    batch.stack()

    batch.unstack()


def test_stack_runtime_error():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch.stack()

    with pytest.raises(RuntimeError):
        batch.append([])


def test_context_stack():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    assert not (batch.is_stacked())
    with batch.stacked_mode():
        assert batch.is_stacked()

    assert not (batch.is_stacked())


def test_context_not_stack():
    class Image(Document):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch.stack()

    assert batch.is_stacked()
    with batch.unstacked_mode():
        assert not (batch.is_stacked())

    assert batch.is_stacked()
