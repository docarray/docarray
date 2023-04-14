import numpy as np

from docarray import BaseDoc
from docarray.array.doc_vec.doc_vec import DocVec
from docarray.typing import AnyTensor, NdArray


def test_da_init():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros(10), name='hello') for _ in range(4)]

    da = DocVec[MyDoc](docs, tensor_type=NdArray)

    assert (da._storage.tensor_columns['tensor'] == np.zeros((4, 10))).all()
    assert da._storage.any_columns['name']._data == ['hello' for _ in range(4)]


def test_da_iter():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=i * np.zeros((10, 10)), name=f'hello{i}') for i in range(4)]

    da = DocVec[MyDoc](docs, tensor_type=NdArray)

    for i, doc in enumerate(da):
        assert isinstance(doc, MyDoc)
        assert (doc.tensor == i * np.zeros((10, 10))).all()
        assert doc.name == f'hello{i}'
