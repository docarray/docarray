from typing import (
    TYPE_CHECKING,
    Sequence,
)

import numpy as np

from ...helper import typename

if TYPE_CHECKING:
    from ...typing import (
        DocumentArrayIndexType,
    )


class DelItemMixin:
    """Provide help function to enable advanced indexing in `__delitem__`"""

    def __delitem__(self, index: 'DocumentArrayIndexType'):
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            self._del_doc_by_offset(int(index))

        elif isinstance(index, str):
            if index.startswith('@'):
                raise NotImplementedError(
                    'Delete elements along traversal paths is not implemented'
                )
            else:
                self._del_doc(index)
        elif isinstance(index, slice):
            self._del_docs_by_slice(index)
        elif index is Ellipsis:
            self._del_all_docs()
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
                # TODO: add support for cases such as da[1, ['text', 'id']]?
                if isinstance(index[0], (str, int)) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self:
                        del self[index[0]]
                        del self[index[1]]
                    else:
                        self._set_doc_attr_by_id(index[0], index[1], None)
                elif isinstance(index[0], (slice, Sequence)):
                    _attrs = index[1]
                    if isinstance(_attrs, str):
                        _attrs = (index[1],)
                    for _d in self[index[0]]:
                        for _aa in _attrs:
                            self._set_doc_attr_by_id(_d.id, _aa, None)
                            _d.pop(_aa)

            elif isinstance(index[0], bool):
                self._del_docs_by_mask(index)
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
