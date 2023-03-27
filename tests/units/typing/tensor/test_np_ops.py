import numpy as np

from docarray import BaseDoc
from docarray.typing import NdArray


def test_tensor_ops():
    class A(BaseDoc):
        tensor: NdArray[3, 224, 224]

    class B(BaseDoc):
        tensor: NdArray[3, 112, 224]

    tensor = A(tensor=np.ones((3, 224, 224))).tensor
    tensord = A(tensor=np.ones((3, 224, 224))).tensor
    tensorn = np.zeros((3, 224, 224))
    tensorhalf = B(tensor=np.ones((3, 112, 224))).tensor
    tensorfull = np.concatenate([tensorhalf, tensorhalf], axis=1)

    assert type(tensor) == NdArray
    assert type(tensor + tensord) == NdArray
    assert type(tensor + tensorn) == NdArray
    assert type(tensor + tensorfull) == NdArray
