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
    overload,
    Any,
    List,
)

import numpy as np

from .mixins import AllMixins
from .. import Document
from ..helper import typename

if TYPE_CHECKING:
    from ..types import (
        DocumentArraySourceType,
        DocumentArrayIndexType,
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
        DocumentArrayMultipleAttributeType,
        DocumentArraySingleAttributeType,
    )


class DocumentArray(AllMixins, MutableSequence[Document]):
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

    @overload
    def __getitem__(self, index: 'DocumentArraySingletonIndexType') -> 'Document':
        ...

    @overload
    def __getitem__(self, index: 'DocumentArrayMultipleIndexType') -> 'DocumentArray':
        ...

    @overload
    def __getitem__(self, index: 'DocumentArraySingleAttributeType') -> List[Any]:
        ...

    @overload
    def __getitem__(
        self, index: 'DocumentArrayMultipleAttributeType'
    ) -> List[List[Any]]:
        ...

    def __getitem__(
        self, index: 'DocumentArrayIndexType'
    ) -> Union['Document', 'DocumentArray']:
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            return self._data[int(index)]
        elif isinstance(index, str):
            if index.startswith('@'):
                return self.traverse_flat(index[1:])
            else:
                return self._data[self._id2offset[index]]
        elif isinstance(index, slice):
            return DocumentArray(self._data[index])
        elif index is Ellipsis:
            return self.flatten()
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self._id2offset:
                        return DocumentArray([self[index[0]], self[index[1]]])
                    else:
                        return getattr(self[index[0]], index[1])
                elif isinstance(index[0], (slice, Sequence)):
                    _docs = self[index[0]]
                    _attrs = index[1]
                    if isinstance(_attrs, str):
                        _attrs = (index[1],)
                    return _docs._get_attributes(*_attrs)
            elif isinstance(index[0], bool):
                return DocumentArray(itertools.compress(self._data, index))
            elif isinstance(index[0], int):
                return DocumentArray(self._data[t] for t in index)
            elif isinstance(index[0], str):
                return DocumentArray(self._data[self._id2offset[t]] for t in index)
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                return self[index.tolist()]
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    @overload
    def __setitem__(
        self,
        index: 'DocumentArrayMultipleAttributeType',
        value: List[List['Any']],
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArraySingleAttributeType',
        value: List['Any'],
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArraySingletonIndexType',
        value: 'Document',
    ):
        ...

    @overload
    def __setitem__(
        self,
        index: 'DocumentArrayMultipleIndexType',
        value: Sequence['Document'],
    ):
        ...

    def __setitem__(
        self,
        index: 'DocumentArrayIndexType',
        value: Union['Document', Sequence['Document']],
    ):

        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            index = int(index)
            self._data[index] = value
            self._id2offset[value.id] = index
        elif isinstance(index, str):
            if index.startswith('@'):
                for _d, _v in zip(self.traverse_flat(index[1:]), value):
                    _d._data = _v._data
                self._rebuild_id2offset()
            else:
                old_idx = self._id2offset.pop(index)
                self._data[old_idx] = value
                self._id2offset[value.id] = old_idx
        elif isinstance(index, slice):
            self._data[index] = value
            self._rebuild_id2offset()
        elif index is Ellipsis:
            for _d, _v in zip(self.flatten(), value):
                _d._data = _v._data
            self._rebuild_id2offset()
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self._id2offset:
                        for _d, _v in zip((self[index[0]], self[index[1]]), value):
                            _d._data = _v._data
                        self._rebuild_id2offset()
                    elif hasattr(self[index[0]], index[1]):
                        setattr(self[index[0]], index[1], value)
                    else:
                        # to avoid accidentally add new unsupport attribute
                        raise ValueError(
                            f'`{index[1]}` is neither a valid id nor attribute name'
                        )
                elif isinstance(index[0], (slice, Sequence)):
                    _docs = self[index[0]]
                    _attrs = index[1]

                    if isinstance(_attrs, str):
                        # a -> [a]
                        # [a, a] -> [a, a]
                        _attrs = (index[1],)
                    if isinstance(value, (list, tuple)) and not any(
                        isinstance(el, (tuple, list)) for el in value
                    ):
                        # [x] -> [[x]]
                        # [[x], [y]] -> [[x], [y]]
                        value = (value,)
                    if not isinstance(value, (list, tuple)):
                        # x -> [x]
                        value = (value,)

                    for _a, _v in zip(_attrs, value):
                        if _a == 'tensor':
                            _docs.tensors = _v
                        elif _a == 'embedding':
                            _docs.embeddings = _v
                        else:
                            if len(_docs) == 1:
                                setattr(_docs[0], _a, _v)
                            else:
                                for _d, _vv in zip(_docs, _v):
                                    setattr(_d, _a, _vv)
            elif isinstance(index[0], bool):
                if len(index) != len(self._data):
                    raise IndexError(
                        f'Boolean mask index is required to have the same length as {len(self._data)}, '
                        f'but receiving {len(index)}'
                    )
                _selected = itertools.compress(self._data, index)
                for _idx, _val in zip(_selected, value):
                    self[_idx.id] = _val
            elif isinstance(index[0], (int, str)):
                if not isinstance(value, Sequence) or len(index) != len(value):
                    raise ValueError(
                        f'Number of elements for assigning must be '
                        f'the same as the index length: {len(index)}'
                    )
                if isinstance(value, Document):
                    for si in index:
                        self[si] = value
                else:
                    for si, _val in zip(index, value):
                        self[si] = _val
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                self[index.tolist()] = value
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def __delitem__(self, index: 'DocumentArrayIndexType'):
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            index = int(index)
            self._id2offset.pop(self._data[index].id)
            del self._data[index]
        elif isinstance(index, str):
            if index.startswith('@'):
                raise NotImplementedError(
                    'Delete elements along traversal paths is not implemented'
                )
            else:
                del self._data[self._id2offset[index]]
            self._id2offset.pop(index)
        elif isinstance(index, slice):
            del self._data[index]
            self._rebuild_id2offset()
        elif index is Ellipsis:
            self._data.clear()
            self._id2offset.clear()
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self._id2offset:
                        del self[index[0]]
                        del self[index[1]]
                    else:
                        self[index[0]].pop(index[1])
                elif isinstance(index[0], (slice, Sequence)):
                    _docs = self[index[0]]
                    _attrs = index[1]
                    if isinstance(_attrs, str):
                        _attrs = (index[1],)
                    for _d in _docs:
                        _d.pop(*_attrs)
            elif isinstance(index[0], bool):
                self._data = list(
                    itertools.compress(self._data, (not _i for _i in index))
                )
                self._rebuild_id2offset()
            elif isinstance(index[0], int):
                for t in sorted(index, reverse=True):
                    del self[t]
            elif isinstance(index[0], str):
                for t in index:
                    del self[t]
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                del self[index.tolist()]
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

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
