import numpy as np

from docarray import BaseDocument
from docarray.array.stacked.array_stacked import DocumentArrayStacked
from docarray.typing import AnyTensor, NdArray


def test_da_init():
    class MyDoc(BaseDocument):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros(10), name='hello') for _ in range(4)]

    da = DocumentArrayStacked[MyDoc](docs, tensor_type=NdArray)

    assert (da._storage.tensor_storage['tensor'] == np.zeros((4, 10))).all()
    assert da._storage.any_storage['name'] == ['hello' for _ in range(4)]
