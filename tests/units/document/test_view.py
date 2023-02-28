import numpy as np

from docarray import BaseDocument
from docarray.array.stacked.column_storage import ColumnStorage, ColumnStorageView
from docarray.typing import AnyTensor, NdArray


def test_document_view():
    class MyDoc(BaseDocument):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=i) for i in range(4)]

    storage = ColumnStorage(docs, MyDoc, NdArray)

    doc = MyDoc.from_view(ColumnStorageView(0, storage))
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
