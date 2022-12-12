import numpy as np
import torch

from docarray import Document, DocumentArray
from docarray.typing import NdArray, TorchTensor


def test_proto_stacked_mode_torch():
    class MyDoc(Document):
        tensor: TorchTensor[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    da.stack()

    da.from_protobuf(da.to_protobuf())


def test_proto_stacked_mode_numpy():
    class MyDoc(Document):
        tensor: NdArray[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    da.stack()

    da.from_protobuf(da.to_protobuf())
