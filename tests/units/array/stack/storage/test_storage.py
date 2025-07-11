import numpy as np

from docarray import BaseDoc
from docarray.array import DocVec
from docarray.array.doc_vec.column_storage import ColumnStorageView
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

    storage = DocVec[MyDoc](docs)._storage

    assert (storage.tensor_columns['tensor'] == np.zeros((4, 10))).all()
    for name in storage.any_columns['name']:
        assert name == 'hello'
    inner_docs = storage.doc_columns['doc']
    assert isinstance(inner_docs, DocVec)
    for i, doc in enumerate(inner_docs):
        assert isinstance(doc, InnerDoc)
        assert doc.price == i


def test_column_storage_view():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=str(i)) for i in range(4)]

    storage = DocVec[MyDoc](docs)._storage

    view = ColumnStorageView(0, storage)

    assert view['id'] == '0'
    assert (view['tensor'] == np.zeros(10)).all()
    assert view['name'] == 'hello'

    view['id'] = '1'
    view['tensor'] = np.ones(10)
    view['name'] = 'byebye'

    assert storage.any_columns['id'][0] == '1'
    assert (storage.tensor_columns['tensor'][0] == np.ones(10)).all()
    assert storage.any_columns['name'][0] == 'byebye'


def test_column_storage_to_dict():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=str(i)) for i in range(4)]

    storage = DocVec[MyDoc](docs)._storage

    view = ColumnStorageView(0, storage)

    dict_view = view.to_dict()

    assert dict_view['id'] == '0'
    assert (dict_view['tensor'] == np.zeros(10)).all()
    assert np.may_share_memory(dict_view['tensor'], view['tensor'])
    assert dict_view['name'] == 'hello'


def test_storage_view_dict_like():
    class MyDoc(BaseDoc):
        tensor: AnyTensor
        name: str

    docs = [MyDoc(tensor=np.zeros((10, 10)), name='hello', id=str(i)) for i in range(4)]

    storage = DocVec[MyDoc](docs)._storage

    view = ColumnStorageView(0, storage)

    assert list(view.keys()) == ['id', 'name', 'tensor']

    # since boolean value of np array is ambiguous, we iterate manually
    for val_view, val_reference in zip(view.values(), ['0', 'hello', np.zeros(10)]):
        if isinstance(val_view, np.ndarray):
            assert (val_view == val_reference).all()
        else:
            assert val_view == val_reference
    for item_view, item_reference in zip(
        view.items(), [('id', '0'), ('name', 'hello'), ('tensor', np.zeros(10))]
    ):
        if isinstance(item_view[1], np.ndarray):
            assert item_view[0] == item_reference[0]
            assert (item_view[1] == item_reference[1]).all()
        else:
            assert item_view == item_reference
