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
    from ... import DocumentArray


class GetItemMixin:
    """Provide helper functions to enable advance indexing in `__getitem__`"""

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
                and (
                    isinstance(index[0], (slice, Sequence, str, int))
                    or index[0] is Ellipsis
                )
                and isinstance(index[1], (str, Sequence))
            ):
                # TODO: add support for cases such as da[1, ['text', 'id']]?
                if isinstance(index[0], (str, int)) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self:
                        return DocumentArray([self[index[0]], self[index[1]]])
                    else:
                        _docs = self[index[0]]
                        if not _docs:
                            return []
                        if isinstance(_docs, Document):
                            return getattr(_docs, index[1])
                        return _docs._get_attributes(index[1])
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
