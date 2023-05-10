import numpy as np

from docarray import BaseDoc
from docarray.array import DocVec
from docarray.array.doc_vec.column_storage import ColumnStorageView
from docarray.typing import AnyTensor


def test_document_view():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=i) for i in range(4)]

    doc_vec = DocVec[MyDoc](docs)
    storage = doc_vec._storage

    result = str(doc_vec[0])
    assert 'MyDoc' in result
    assert 'id' in result
    assert 'tensor' in result
    assert 'name' in result

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
