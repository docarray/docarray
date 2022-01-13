import itertools
import uuid
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
import weaviate

from .mixins import AllMixins
from .. import Document, DocumentArray
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


class DocumentArrayWeaviate(AllMixins, MutableSequence[Document]):
    def __init__(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        array_id: str = None,
    ):
        super().__init__()
        self._client = weaviate.Client('http://localhost:8080')
        if array_id:
            self._class_name = array_id
        else:
            self._class_name = self._get_weaviate_class_name()
            self._upload_weaviate_schema()
        self._offset2ids = []
        if docs is None:
            return
        elif isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            self.extend(Document(d, copy=True) for d in docs)
        else:
            if isinstance(docs, Document):
                self.append(docs)

    def _get_weaviate_class_name(self):
        return ''.join([i for i in uuid.uuid1().hex if not i.isdigit()]).capitalize()

    def _upload_weaviate_schema(self):
        doc_schema = {
            'classes': [
                {
                    'class': self._class_name,
                    'properties': [
                        {'dataType': ['blob'], 'name': '_serialized'},
                    ],
                }
            ]
        }
        self._client.schema.create(doc_schema)

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._offset2ids.insert(index, value.id)
        self._client.data_object.create(**self._doc2weaviate_create_payload(value))

    def _doc2weaviate_create_payload(self, value: 'Document'):
        return dict(
            data_object={'_serialized': value.to_base64()},
            class_name=self._class_name,
            uuid=str(uuid.UUID(value.id)),
            vector=value.embedding or [0],
        )

    def __eq__(self, other):
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._client.get_meta() == other._client.get_meta()
        )

    def __len__(self):
        return (
            self._client.query.aggregate(self._class_name)
            .with_meta_count()
            .do()['data']['Aggregate'][self._class_name][0]['meta']['count']
        )

    def __iter__(self) -> Iterator['Document']:
        for _id in self._offset2ids:
            yield self[_id]

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._client.data_object.exists(x)
        elif isinstance(x, Document):
            return self._client.data_object.exists(x.id)
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
            # convert into integer, map to string key, delegate the string __getitem__
            return self[self._offset2ids[int(index)]]
        elif isinstance(index, str):
            if index.startswith('@'):
                return self.traverse_flat(index[1:])
            else:
                return Document.from_base64(
                    self._client.data_object.get_by_id(index, with_vector=True)[
                        'properties'
                    ]['_serialized']
                )
        elif isinstance(index, slice):
            # convert it to a sequence of strings.
            _ids = self._offset2ids[index]
            return DocumentArray(self[_id] for _id in _ids)
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
                    if index[1] in self._offset2ids:
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
                _ids = itertools.compress(self._offset2ids, index)
                return self[_ids]
            elif isinstance(index[0], int):
                return DocumentArray(self[t] for t in index)
            elif isinstance(index[0], str):
                return DocumentArray(self[t] for t in index)
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
                        if _a == 'blob':
                            _docs.blobs = _v
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
        self._offset2ids.clear()

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def __repr__(self):
        return f'<{self.__class__.__name__} (length={len(self)}) at {id(self)}>'

    def __add__(self, other: 'Document'):
        v = type(self)()
        for doc in self:
            v.append(doc)
        for doc in other:
            v.append(doc)
        return v

    def extend(self, values: Iterable['Document']) -> None:
        with self._client.batch as _b:
            for d in values:
                _b.add_data_object(**self._doc2weaviate_create_payload(d))
                self._offset2ids.append(d.id)
