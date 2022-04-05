import itertools
from typing import (
    TYPE_CHECKING,
    Union,
    Sequence,
    overload,
    Any,
    List,
)

import numpy as np

from ... import Document
from ...helper import typename

if TYPE_CHECKING:
    from ...typing import (
        DocumentArrayIndexType,
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
        DocumentArrayMultipleAttributeType,
        DocumentArraySingleAttributeType,
    )


class SetItemMixin:
    """Provides helper function to allow advanced indexing for `__setitem__`"""

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

        # set by offset
        # allows da[1] = Document()
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            self._set_doc_by_offset(int(index), value)
        elif isinstance(index, str):
            # set by traversal paths
            # allows da['@m,c] = [m1, m2, ..., mn, c1, c2, ..., cp]
            if index.startswith('@'):
                self._set_doc_value_pairs_nested(self.traverse_flat(index[1:]), value)

            # set by ID
            # allows da['id_123'] = Document()
            else:
                self._set_doc(index, value)
        # set by slice
        # allows da[1:3] = [d1, d2]
        elif isinstance(index, slice):
            self._set_docs_by_slice(index, value)

        # flatten and set
        # allows da[...] = [d1, d2,..., dn]
        elif index is Ellipsis:
            self._set_doc_value_pairs(self.flatten(), value)

        # index is sequence
        elif isinstance(index, Sequence):
            # allows da[idx1, idx2] = value
            if isinstance(index, tuple) and len(index) == 2:
                self._set_by_pair(index[0], index[1], value)

            # allows da[True, False, True, True]
            elif isinstance(index[0], bool):
                self._set_by_mask(index, value)

            # allows da[id1, id2, id3] = [d1, d2, d3]
            elif isinstance(index[0], (int, str)):
                for si, _val in zip(index, value):
                    self[si] = _val  # leverage existing setter
            else:
                raise IndexError(
                    f'{index} should be either a sequence of bool, int or str'
                )

        # set by ndarray
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                self[index.tolist()] = value  # leverage existing setter
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def _set_by_pair(self, idx1, idx2, value):

        if isinstance(idx1, str) and not idx1.startswith('@'):
            # second is an ID
            # allows da[id1, id2] = [d1, d2]
            if isinstance(idx2, str) and idx2 in self:
                self._set_doc_value_pairs((self[idx1], self[idx2]), value)
            # second is an attribute
            # allows da[id, attr] = attr_value
            elif isinstance(idx2, str) and hasattr(self[idx1], idx2):
                self._set_doc_attr_by_id(idx1, idx2, value)
            # second is a list of attributes:
            # allows da[id, [attr1, attr2, attr3]] = [v1, v2, v3]
            elif (
                isinstance(idx2, Sequence)
                and all(isinstance(attr, str) for attr in idx2)
                and all(hasattr(self[idx1], attr) for attr in idx2)
            ):

                for attr, _v in zip(idx2, value):
                    self._set_doc_attr_by_id(idx1, attr, _v)
            else:
                raise IndexError(f'`{idx2}` is neither a valid id nor attribute name')
        elif isinstance(idx1, int):
            # second is an offset
            # allows da[offset1, offset2] = [d1, d2]
            if isinstance(idx2, int):
                self._set_doc_value_pairs((self[idx1], self[idx2]), value)
            # second is an attribute
            # allows da[offset, attr] = value
            elif isinstance(idx2, str) and hasattr(self[idx1], idx2):
                self._set_doc_attr_by_offset(idx1, idx2, value)
            # second is a list of attributes
            # allows da[offset, [attr1, attr2, attr3]] = [v1, v2, v3]
            elif (
                isinstance(idx2, Sequence)
                and all(isinstance(attr, str) for attr in idx2)
                and all(hasattr(self[idx1], attr) for attr in idx2)
            ):
                for attr, _v in zip(idx2, value):
                    self._set_doc_attr_by_offset(idx1, attr, _v)
            else:
                raise IndexError(f'`{idx2}` must be an attribute or list of attributes')

        # allows da[sequence/slice/ellipsis/traversal_path, attributes] = [v1, v2, ...]
        elif (
            isinstance(idx1, (slice, Sequence))
            or idx1 is Ellipsis
            or (isinstance(idx1, str) and idx1.startswith('@'))
        ):
            self._set_docs_attributes(idx1, idx2, value)
        else:
            raise IndexError(f'Unsupported first index type {typename(idx1)}: {idx1}')

    def _set_by_mask(self, mask: List[bool], value):
        _selected = itertools.compress(self, mask)
        self._set_doc_value_pairs(_selected, value)

    def _set_docs_attributes(self, index, attributes, value):
        if isinstance(attributes, str):
            # a -> [a]
            # [a, a] -> [a, a]
            attributes = (attributes,)
            value = (value,)

        if isinstance(index, str) and index.startswith('@'):
            self._set_docs_attributes_traversal_paths(index, attributes, value)
        elif index is Ellipsis:
            _docs = self[index]
            for _a, _v in zip(attributes, value):
                if _a == 'tensor':
                    _docs.tensors = _v
                elif _a == 'embedding':
                    _docs.embeddings = _v
                else:
                    if not isinstance(_v, (list, tuple)):
                        for _d in _docs:
                            setattr(_d, _a, _v)
                    else:
                        for _d, _vv in zip(_docs, _v):
                            setattr(_d, _a, _vv)
            self._set_doc_value_pairs_nested(_docs, _docs)
        else:
            _docs = self[index]
            if not _docs:
                return

            for _a, _v in zip(attributes, value):
                if _a in ('tensor', 'embedding'):
                    if _a == 'tensor':
                        _docs.tensors = _v
                    elif _a == 'embedding':
                        _docs.embeddings = _v
                    for _d in _docs:
                        self._set_doc(_d.id, _d)
                else:
                    if not isinstance(_v, (list, tuple)):
                        for _d in _docs:
                            self._set_doc_attr_by_id(_d.id, _a, _v)
                    else:
                        for _d, _vv in zip(_docs, _v):
                            self._set_doc_attr_by_id(_d.id, _a, _vv)

    def _set_docs_attributes_traversal_paths(
        self, traversal_paths: str, attributes, value
    ):
        _docs = self[traversal_paths]
        if not _docs:
            return

        for _a, _v in zip(attributes, value):
            if _a == 'tensor':
                _docs.tensors = _v
            elif _a == 'embedding':
                _docs.embeddings = _v
            else:
                if not isinstance(_v, (list, tuple)):
                    for _d in _docs:
                        setattr(_d, _a, _v)
                else:
                    for _d, _vv in zip(_docs, _v):
                        setattr(_d, _a, _vv)
        self._set_doc_value_pairs_nested(_docs, _docs)
