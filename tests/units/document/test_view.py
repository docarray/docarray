import re

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

    delimiters = [",", '(', '=', ')']
    result = re.split('|'.join(map(re.escape, delimiters)), str(doc_vec[0]))
    assert (
        re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?)?[m|K]?', '', result[0]) == 'MyDoc'
    )
    assert re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?)?[m|K]?', '', result[1]) == 'id'
    assert re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?)?[m|K]?', '', result[2]) == '0'

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
