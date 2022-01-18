import itertools
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Union,
    Sequence,
    overload,
    Any,
    List,
    Iterable,
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
    from ... import DocumentArray


class GetItemMixin:
    """Provide helper functions to enable advance indexing in `__getitem__`"""

    @abstractmethod
    def _get_doc_by_offset(self, offset: int) -> 'Document':
        ...

    @abstractmethod
    def _get_doc_by_id(self, _id: str) -> 'Document':
        ...

    @abstractmethod
    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_offset(j) for j in range(len(self))[_slice])

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_offset(d) for d in offsets)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_id`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_id(d) for d in ids)

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
            return self._get_doc_by_offset(int(index))
        elif isinstance(index, str):
            if index.startswith('@'):
                return self.traverse_flat(index[1:])
            else:
                return self._get_doc_by_id(index)
        elif isinstance(index, slice):
            from ... import DocumentArray

            return DocumentArray(self._get_docs_by_slice(index))
        elif index is Ellipsis:
            return self.flatten()
        elif isinstance(index, Sequence):
            from ... import DocumentArray

            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self:
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
                return DocumentArray(itertools.compress(self, index))
            elif isinstance(index[0], int):
                return DocumentArray(self._get_docs_by_offsets(index))
            elif isinstance(index[0], str):
                return DocumentArray(self._get_docs_by_ids(index))
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                return self[index.tolist()]
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        raise IndexError(f'Unsupported index type {typename(index)}: {index}')
