import numpy as np
import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.array import DocumentArrayStacked
from docarray.typing import NdArray, TorchTensor


@pytest.fixture()
def batch():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    return batch.stack()


@pytest.mark.proto
def test_proto_stacked_mode_torch(batch):
    batch.from_protobuf(batch.to_protobuf())


@pytest.mark.proto
def test_proto_stacked_mode_numpy():
    class MyDoc(BaseDocument):
        tensor: NdArray[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    da = da.stack()

    da.from_protobuf(da.to_protobuf())


@pytest.mark.proto
def test_stacked_proto():
    class CustomDocument(BaseDocument):
        image: NdArray

    da = DocumentArray[CustomDocument](
        [CustomDocument(image=np.zeros((3, 224, 224))) for _ in range(10)]
    ).stack()

    da2 = DocumentArrayStacked.from_protobuf(da.to_protobuf())

    assert isinstance(da2, DocumentArrayStacked)
