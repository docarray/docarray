from abc import ABC

import pytest

from docarray import DocumentArray, Document
from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.memory import BackendMixin, SequenceLikeMixin


class DummyGetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essentials

    def _del_doc_by_id(self, _id: str):
        del self._data[_id]

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            del self._data[_id]
        self._data[value.id] = value

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[_id]

    def _clear_storage(self):
        self._data.clear()


class StorageMixins(BackendMixin, DummyGetSetDelMixin, SequenceLikeMixin, ABC):
    ...


class DocumentArrayDummy(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def _load_offset2ids(self):
        self._offset2ids = Offset2ID()

    def _save_offset2ids(self):
        pass


@pytest.fixture(scope='function')
def docs():
    return DocumentArrayDummy([Document(id=str(j), text=str(j)) for j in range(100)])


def test_index_by_int_str(docs):
    # getter
    assert len(docs[[1]]) == 1
    assert len(docs[1, 2]) == 2
    assert len(docs[1, 2, 3]) == 3
    assert len(docs[1:5]) == 4
    assert len(docs[1:100:5]) == 20  # 1 to 100, sep with 5

    # setter
    with pytest.raises(TypeError, match='an iterable'):
        docs[1:5] = Document(text='repl')

    docs[1:5] = [Document(text=f'repl{j}') for j in range(4)]
    for d in docs[1:5]:
        assert d.text.startswith('repl')
    assert len(docs) == 100


def test_getter_int_str(docs):
    # getter
    assert docs[99].text == '99'
    assert docs[-1].text == '99'
    assert docs[0].text == '0'

    # string index
    assert docs['0'].text == '0'
    assert docs['99'].text == '99'

    with pytest.raises(IndexError):
        docs[100]

    with pytest.raises(KeyError):
        docs['adsad']


def test_set_content_none():
    da = DocumentArray(
        [
            Document(mime_type='image'),
            Document(mime_type='image'),
            Document(mime_type='text'),
        ]
    )

    txt_da = da.find({'mime_type': {'$eq': 'image'}})
    assert len(txt_da) == 2
    txt_da.texts = ['hello', 'world']
    assert txt_da.texts == ['hello', 'world']
    assert da.texts == ['hello', 'world', '']
    da.tensors = None
    assert da.texts == ['hello', 'world', '']
