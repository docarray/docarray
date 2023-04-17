import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

import numpy as np

from docarray.utils._internal.misc import import_library

T_item = TypeVar('T_item')
T = TypeVar('T', bound='IndexingSequenceMixin')

IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


def _is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it


class IndexingSequenceMixin(Iterable[T_item]):
    """
    This mixin allow sto extend a list into an object that can be indexed
    a la numpy/pytorch.

    You can index into, delete from, and set items in a IndexingSequenceMixin like a numpy doc_list or torch tensor:

    ---

    ```python
    docs[0]  # index by position
    docs[0:5:2]  # index by slice
    docs[[0, 2, 3]]  # index by list of indices
    docs[True, False, True, True, ...]  # index by boolean mask
    ```

    ---

    """

    _data: MutableSequence[T_item]

    @abc.abstractmethod
    def __init__(
        self,
        docs: Optional[Iterable[T_item]] = None,
    ):
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def _normalize_index_item(
        item: Any,
    ) -> Union[int, slice, Iterable[int], Iterable[bool], None]:
        # basic index types
        if item is None or isinstance(item, (int, slice, tuple, list)):
            return item

        # numpy index types
        if _is_np_int(item):
            return item.item()

        index_has_getitem = hasattr(item, '__getitem__')
        is_valid_bulk_index = index_has_getitem and isinstance(item, Iterable)
        if not is_valid_bulk_index:
            raise ValueError(f'Invalid index type {type(item)}')

        if isinstance(item, np.ndarray) and (
            item.dtype == np.bool_ or np.issubdtype(item.dtype, np.integer)
        ):
            return item.tolist()

        # torch index types
        if TYPE_CHECKING:
            import torch
        else:
            torch = import_library('torch', raise_error=True)

        allowed_torch_dtypes = [
            torch.bool,
            torch.int64,
        ]
        if isinstance(item, torch.Tensor) and (item.dtype in allowed_torch_dtypes):
            return item.tolist()

        return item

    def _get_from_indices(self: T, item: Iterable[int]) -> T:
        results = []
        for ix in item:
            results.append(self._data[ix])
        return self.__class__(results)

    def _set_by_indices(self: T, item: Iterable[int], value: Iterable[T_item]):
        for ix, doc_to_set in zip(item, value):
            try:
                self._data[ix] = doc_to_set
            except KeyError:
                raise IndexError(f'Index {ix} is out of range')

    def _get_from_mask(self: T, item: Iterable[bool]) -> T:
        return self.__class__(
            [doc for doc, mask_value in zip(self, item) if mask_value]
        )

    def _set_by_mask(self: T, item: Iterable[bool], value: Sequence[T_item]):
        i_value = 0
        for i, mask_value in zip(range(len(self)), item):
            if mask_value:
                self._data[i] = value[i_value]
                i_value += 1

    def _del_from_mask(self: T, item: Iterable[bool]) -> None:
        idx_to_delete = [i for i, val in enumerate(item) if val]
        self._del_from_indices(idx_to_delete)

    def _del_from_indices(self: T, item: Iterable[int]) -> None:
        for ix in sorted(item, reverse=True):
            # reversed is needed here otherwise some the indices are not up to date after
            # each delete
            del self._data[ix]

    def __delitem__(self, key: Union[int, IndexIterType]) -> None:
        item = self._normalize_index_item(key)

        if item is None:
            return
        elif isinstance(item, (int, slice)):
            del self._data[item]
        else:
            head = item[0]  # type: ignore
            if isinstance(head, bool):
                item_ = cast(Iterable[bool], item)
                return self._del_from_mask(item_)
            elif isinstance(head, int):
                return self._del_from_indices(item)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')

    @overload
    def __getitem__(self: T, item: int) -> T_item:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item):
        item = self._normalize_index_item(item)

        if type(item) == slice:
            return self.__class__(self._data[item])

        if isinstance(item, int):
            return self._data[item]

        if item is None:
            return self

        # _normalize_index_item() guarantees the line below is correct
        head = item[0]  # type: ignore
        if isinstance(head, bool):
            return self._get_from_mask(item)
        elif isinstance(head, int):
            return self._get_from_indices(item)
        else:
            raise TypeError(f'Invalid type {type(head)} for indexing')

    @overload
    def __setitem__(self: T, key: IndexIterType, value: Sequence[T_item]):
        ...

    @overload
    def __setitem__(self: T, key: int, value: T_item):
        ...

    @no_type_check
    def __setitem__(self: T, key, value):
        key_norm = self._normalize_index_item(key)

        if isinstance(key_norm, int):
            self._data[key_norm] = value
        elif isinstance(key_norm, slice):
            self._data[key_norm] = value
        else:
            # _normalize_index_item() guarantees the line below is correct
            head = key_norm[0]
            if isinstance(head, bool):
                return self._set_by_mask(key_norm, value)
            elif isinstance(head, int):
                return self._set_by_indices(key_norm, value)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')
