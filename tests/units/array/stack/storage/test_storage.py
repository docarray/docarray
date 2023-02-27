import numpy as np

from docarray import BaseDocument
from docarray.array.stacked.storage import Storage, StorageView
from docarray.typing import AnyTensor, NdArray


def test_storage_init():
    class MyDoc(BaseDocument):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros(10), name='hello') for _ in range(4)]

    storage = Storage(docs, MyDoc, NdArray)

    assert (storage.tensor_storage['tensor'] == np.zeros((4, 10))).all()
    assert storage.any_storage['name'] == ['hello' for _ in range(4)]


def test_storage_view():
    class MyDoc(BaseDocument):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=i) for i in range(4)]

    storage = Storage(docs, MyDoc, NdArray)

    view = StorageView(0, storage)

    data = view.data()

    assert data['id'] == '0'
    assert (data['tensor'] == np.zeros(10)).all()
    assert data['name'] == 'hello'
