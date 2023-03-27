import numpy as np

from docarray import BaseDoc
from docarray.array import DocArrayStacked
from docarray.array.stacked.column_storage import ColumnStorageView
from docarray.typing import AnyTensor


def test_column_storage_init():
    class InnerDoc(BaseDoc):
        price: int

    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str
        doc: InnerDoc

    docs = [
        MyDoc(tensor=np.zeros(10), name='hello', doc=InnerDoc(price=i))
        for i in range(4)
    ]

    storage = DocArrayStacked[MyDoc](docs)._storage

    assert (storage.tensor_columns['tensor'] == np.zeros((4, 10))).all()
    for name in storage.any_columns['name']:
        assert name == 'hello'
    inner_docs = storage.doc_columns['doc']
    assert isinstance(inner_docs, DocArrayStacked[InnerDoc])
    for i, doc in enumerate(inner_docs):
        assert doc.price == i


def test_column_storage_view():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=i) for i in range(4)]

    storage = DocArrayStacked[MyDoc](docs)._storage

    view = ColumnStorageView(0, storage)

    assert view['id'] == '0'
    assert (view['tensor'] == np.zeros(10)).all()
    assert view['name'] == 'hello'

    view['id'] = 1
    view['tensor'] = np.ones(10)
    view['name'] = 'byebye'

    assert storage.any_columns['id'][0] == 1
    assert (storage.tensor_columns['tensor'][0] == np.ones(10)).all()
    assert storage.any_columns['name'][0] == 'byebye'
