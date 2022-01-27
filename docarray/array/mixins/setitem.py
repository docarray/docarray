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
    from ...types import (
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

        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            self._set_doc_by_offset(int(index), value)
        elif isinstance(index, str):
            if index.startswith('@'):
                self._set_doc_value_pairs(self.traverse_flat(index[1:]), value)
            else:
                self._set_doc_by_id(index, value)
        elif isinstance(index, slice):
            # TODO: handle this case da[1:3] = DocumentAray.empty(10)
            # where length of value is longer than the slice
            self._set_docs_by_slice(index, value)
        elif index is Ellipsis:
            self._set_doc_value_pairs(self.flatten(), value)
        elif isinstance(index, Sequence):
            if (
                isinstance(index, tuple)
                and len(index) == 2
                and (
                    isinstance(index[0], (slice, Sequence, str, int))
                    or index[0] is Ellipsis
                )
                and isinstance(index[1], (str, Sequence))
            ):
                # TODO: this is added because we are still trying to figure out the proper way
                # to set attribute and to get test_path_syntax_indexing_set to pass.
                # we may have to refactor the following logic

                # NOTE: this check is not proper way to handle, but a temporary hack.
                # writing it this way to minimize effect on other docarray classs and
                # to make it easier to remove/refactor the following block
                if self.__class__.__name__ in {
                    'DocumentArrayWeaviate',
                    'DocumentArrayInMemory',
                }:
                    from ..memory import DocumentArrayInMemory

                    if index[1] in self:
                        # we first handle the case when second item in index is an id not attr
                        _docs = DocumentArrayInMemory(
                            self[index[0]]
                        ) + DocumentArrayInMemory(self[index[1]])
                        self._set_doc_value_pairs(_docs, value)
                        return

                    _docs = self[index[0]]

                    if not _docs:
                        return

                    if isinstance(_docs, Document):
                        _docs = DocumentArrayInMemory(_docs)
                        # because we've augmented docs dimension, we do the same for value
                        value = (value,)

                    attrs = index[1]
                    if isinstance(attrs, str):
                        attrs = (attrs,)
                        # because we've augmented attrs dimension, we do the same for value
                        value = (value,)

                    for attr in attrs:
                        if not hasattr(_docs[0], attr):
                            raise ValueError(
                                f'`{attr}` is neither a valid id nor attribute name'
                            )

                    for _a, _v in zip(attrs, value):
                        self._set_docs_attrs(_docs, _a, _v)
                    return

                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self:
                        self._set_doc_value_pairs(
                            (self[index[0]], self[index[1]]), value
                        )
                    elif hasattr(self[index[0]], index[1]):
                        self._set_doc_attr_by_id(index[0], index[1], value)
                    else:
                        # to avoid accidentally add new unsupport attribute
                        raise ValueError(
                            f'`{index[1]}` is neither a valid id nor attribute name'
                        )
                elif isinstance(index[0], (slice, Sequence)):
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

                    _docs = self[index[0]]
                    for _a, _v in zip(_attrs, value):
                        if _a in ('tensor', 'embedding'):
                            if _a == 'tensor':
                                _docs.tensors = _v
                            elif _a == 'embedding':
                                _docs.embeddings = _v
                            for _d in _docs:
                                self._set_doc_by_id(_d.id, _d)
                        else:
                            if len(_docs) == 1:
                                self._set_doc_attr_by_id(_docs[0].id, _a, _v)
                            else:
                                for _d, _vv in zip(_docs, _v):
                                    self._set_doc_attr_by_id(_d.id, _a, _vv)
            elif isinstance(index[0], bool):
                if len(index) != len(self):
                    raise IndexError(
                        f'Boolean mask index is required to have the same length as {len(self._data)}, '
                        f'but receiving {len(index)}'
                    )
                _selected = itertools.compress(self, index)
                self._set_doc_value_pairs(_selected, value)
            elif isinstance(index[0], (int, str)):
                if not isinstance(value, Sequence) or len(index) != len(value):
                    raise ValueError(
                        f'Number of elements for assigning must be '
                        f'the same as the index length: {len(index)}'
                    )
                if isinstance(value, Document):
                    for si in index:
                        self[si] = value  # leverage existing setter
                else:
                    for si, _val in zip(index, value):
                        self[si] = _val  # leverage existing setter
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
