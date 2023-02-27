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

    assert view['id'] == '0'
    assert (view['tensor'] == np.zeros(10)).all()
    assert view['name'] == 'hello'

    view['id'] = 1
    view['tensor'] = np.ones(10)
    view['name'] = 'byebye'

    assert storage.any_storage['id'][0] == 1
    assert (storage.tensor_storage['tensor'][0] == np.ones(10)).all()
    assert storage.any_storage['name'][0] == 'byebye'


def test_document_view():
    class MyDoc(BaseDocument):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=i) for i in range(4)]

    storage = Storage(docs, MyDoc, NdArray)

    doc = MyDoc.from_view(StorageView(0, storage))
    assert doc.is_view()
    assert doc.id == '0'
    assert (doc.tensor == np.zeros(10)).all()
    assert doc.name == 'hello'

    storage.columns['id'][0] = '12345'
    storage.columns['tensor'][0] = np.ones(10)
    storage.columns['name'][0] = 'byebye'

    assert doc.id == '12345'
    assert (doc.tensor == np.ones(10)).all()
    assert doc.name == 'byebye'
