import itertools
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Iterator,
    Dict,
    Union,
    MutableSequence,
    Sequence,
    Iterable,
    Any,
)

from .mixins import AllMixins
from .. import Document

if TYPE_CHECKING:
    from ..types import (
        DocumentArraySourceType,
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
    )


class DocumentArray(AllMixins, MutableSequence[Document]):
    def _del_docs_by_mask(self, mask: Sequence[bool]):
        self._data = list(itertools.compress(self._data, (not _i for _i in mask)))
        self._rebuild_id2offset()

    def _del_all_docs(self):
        self._data.clear()
        self._id2offset.clear()

    def _del_docs_by_slice(self, _slice: slice):
        del self._data[_slice]
        self._rebuild_id2offset()

    def _del_doc_by_id(self, _id: str):
        del self._data[self._id2offset[_id]]
        self._id2offset.pop(_id)

    def _del_doc_by_offset(self, offset: int):
        self._id2offset.pop(self._data[offset].id)
        del self._data[offset]

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        self._data[offset] = value
        self._id2offset[value.id] = offset

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        old_idx = self._id2offset.pop(_id)
        self._data[old_idx] = value
        self._id2offset[value.id] = old_idx

    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        self._data[_slice] = value
        self._rebuild_id2offset()

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Iterable['Document']
    ):
        for _d, _v in zip(docs, values):
            _d._data = _v._data
        self._rebuild_id2offset()

    def _set_doc_attr_by_index(
        self,
        _index: Union[
            'DocumentArraySingletonIndexType', 'DocumentArrayMultipleIndexType'
        ],
        attr: str,
        value: Any,
    ):
        setattr(self[_index], attr, value)

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        return self._data[offset]

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[self._id2offset[_id]]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._data[_slice]

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        return (self._data[t] for t in offsets)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        return (self._data[self._id2offset[t]] for t in ids)

    def __init__(
        self, docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ):
        super().__init__()
        self._data = []
        if docs is None:
            return
        elif isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            if copy:
                self._data = [Document(d, copy=True) for d in docs]
                self._rebuild_id2offset()
            elif isinstance(docs, DocumentArray):
                self._data = docs._data
                self._id_to_index = docs._id2offset
            else:
                self._data = list(docs)
                self._rebuild_id2offset()
        else:
            if isinstance(docs, Document):
                if copy:
                    self.append(Document(docs, copy=True))
                else:
                    self.append(docs)

    @property
    def _id2offset(self) -> Dict[str, int]:
        """Return the `_id_to_index` map

        :return: a Python dict.
        """
        if not hasattr(self, '_id_to_index'):
            self._rebuild_id2offset()
        return self._id_to_index

    def _rebuild_id2offset(self) -> None:
        """Update the id_to_index map by enumerating all Documents in self._data.

        Very costy! Only use this function when self._data is dramtically changed.
        """

        self._id_to_index = {
            d.id: i for i, d in enumerate(self._data)
        }  # type: Dict[str, int]

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._data.insert(index, value)
        self._id2offset[value.id] = index

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and type(self._data) is type(other._data)
            and self._data == other._data
        )

    def __len__(self):
        return len(self._data)

    def __iter__(self) -> Iterator['Document']:
        yield from self._data

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return x in self._id2offset
        elif isinstance(x, Document):
            return x.id in self._id2offset
        else:
            return False

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._data.clear()
        self._id2offset.clear()

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def __repr__(self):
        return f'<{self.__class__.__name__} (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Sequence['Document']]):
        v = type(self)()
        v.extend(self)
        v.extend(other)
        return v

    def extend(self, values: Iterable['Document']) -> None:
        self._data.extend(values)
        self._rebuild_id2offset()
