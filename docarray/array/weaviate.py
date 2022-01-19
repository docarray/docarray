import itertools
import uuid
from collections.abc import Iterable
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
import scipy.sparse

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


def wmap(doc_id: str):
    # TODO: although rare, it's possible for key collision to occur with SHA-1
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def find(l, item):
    return l.index(item) if item in l else -1


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
        self._client.schema.delete_all()
        self._client.schema.create(doc_schema)

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._offset2ids.insert(index, wmap(value.id))
        self._client.data_object.create(**self._doc2weaviate_create_payload(value))

    def _doc2weaviate_create_payload(self, value: 'Document'):
        if value.embedding is None:
            embedding = [0]
        elif isinstance(value.embedding, scipy.sparse.spmatrix):
            embedding = value.embedding.toarray()
        else:
            embedding = value.embedding

        return dict(
            data_object={'_serialized': value.to_base64()},
            class_name=self._class_name,
            uuid=wmap(value.id),
            vector=embedding,
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
        for int_id in range(len(self._offset2ids)):
            yield self[int_id]

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._client.data_object.exists(wmap(x))
        elif isinstance(x, Document):
            return self._client.data_object.exists(wmap(x.id))
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

    def _getitem(self, wid: str):
        resp = self._client.data_object.get_by_id(wid, with_vector=True)
        if not resp:
            raise KeyError(wid)
        return Document.from_base64(resp['properties']['_serialized'])

    def __getitem__(
        self, index: 'DocumentArrayIndexType'
    ) -> Union['Document', 'DocumentArray']:
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            # convert into integer, map to string key, delegate the string __getitem__
            return self._getitem(self._offset2ids[index])
        elif isinstance(index, str):
            if index.startswith('@'):
                return self.traverse_flat(index[1:])
            else:
                return self._getitem(wmap(index))
        elif isinstance(index, slice):
            # convert it to a sequence of strings.
            _ids = self._offset2ids[index]
            return DocumentArray(self._getitem(_id) for _id in _ids)
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
                    if wmap(index[1]) in self._offset2ids:
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
                _ids = list(itertools.compress(self._offset2ids, index))
                return DocumentArray(self._getitem(_id) for _id in _ids)
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

    def _setitem(self, wid: str, value: Document):
        exist = self._client.data_object.exists(wid)
        self._client.data_object.delete(wid)
        self._client.data_object.create(**self._doc2weaviate_create_payload(value))
        self._offset2ids[find(self._offset2ids, wid)] = wmap(value.id)

    def __setitem__(
        self,
        index: 'DocumentArrayIndexType',
        value: Union['Document', Sequence['Document']],
    ):
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            wid = self._offset2ids[int(index)]
            self._setitem(wid, value)
            # update the weaviatei
            self._offset2ids[index] = wmap(value.id)

        elif isinstance(index, str):
            if index.startswith('@'):
                raise NotImplementedError(
                    'set along traversal paths is not implemented'
                )
            else:
                self._setitem(wmap(index), value)
        elif isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError('can only assign an iterable')
            index = self._offset2ids[index]
            for _i, _v in zip(index, value):
                self._setitem(_i, _v)
        elif index is Ellipsis:
            for _d, _v in zip(self.flatten(), value):
                self._setitem(wmap(d.id), _v)
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    i = find(self._offset2ids, wmap(index[1]))
                    if i >= 0:
                        for _d, _v in zip((self[index[0]], self[index[1]]), value):
                            self._setitem(wmap(_d.id), _v)
                    elif hasattr(self[index[0]], index[1]):
                        doc = self[index[0]]
                        setattr(doc, index[1], value)
                        self._setitem(wmap(doc.id), doc)
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
                        for d in _docs:
                            # TODO: refactor this to batch update
                            self._setitem(wmap(d.id), d)

            elif isinstance(index[0], bool):
                if len(index) != len(self):
                    raise IndexError(
                        f'Boolean mask index is required to have the same length as {len(self)}, '
                        f'but receiving {len(index)}'
                    )
                _selected = itertools.compress(self._offset2ids, index)
                for _idx, _val in zip(_selected, value):
                    self._setitem(_idx, _val)
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

    def _delitem(self, key):
        self._offset2ids.pop(find(self._offset2ids, key))
        self._client.data_object.delete(key)

    def __delitem__(self, index: 'DocumentArrayIndexType'):
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            index = int(index)
            self._delitem(self._offset2ids[index])
        elif isinstance(index, str):
            if index.startswith('@'):
                raise NotImplementedError(
                    'Delete elements along traversal paths is not implemented'
                )
            else:
                self._delitem(wmap(index))
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self)
            step = index.step or 1
            del self[list(range(start, stop, step))]
        elif index is Ellipsis:
            self._client.schema.delete_all()
            self._offset2ids.clear()
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    i = find(self._offset2ids, wmap(index[1]))
                    if i >= 0:
                        del self[index[0]]
                        del self[index[1]]
                    elif index[1] != 'id':
                        doc = self[index[0]]
                        doc.pop(index[1])
                        self._setitem(wmap(index[0]), doc)
                    else:
                        raise ValueError('cannot pop id from DocumentArrayWeaviate')
                elif isinstance(index[0], (slice, Sequence)):
                    _docs = self[index[0]]
                    _attrs = index[1]
                    if 'id' in _attrs:
                        raise ValueError('cannot pop id from DocumentArrayWeaviate')
                    if isinstance(_attrs, str):
                        _attrs = (index[1],)
                    for _d in _docs:
                        _d.pop(*_attrs)
                        self._setitem(wmap(_d.id), _d)
            elif isinstance(index[0], bool):
                idx = list(
                    itertools.compress(self._offset2ids, (not _i for _i in index))
                )
                for _idx in reversed(idx):
                    self._delitem(_idx)
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
        self._client.schema.delete_all()
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
                self._offset2ids.append(wmap(d.id))
