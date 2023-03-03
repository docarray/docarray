import abc
from typing import (
    Any,
    Generic,
    Iterable,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np

from docarray.utils.misc import is_torch_available

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


class IndexingSequenceMixin(Generic[T_item]):
    """
    This mixin allow to extend a list into an object that can be indexed
    a la mnumpy/pytorch.

    You can index into a IndexingSequenceMixin like a numpy array or torch tensor:

    .. code-block:: python
        da[0]  # index by position
        da[0:5:2]  # index by slice
        da[[0, 2, 3]]  # index by list of indices
        da[True, False, True, True, ...]  # index by boolean mask

    You can delete items from a DocumentArray like a Python List

    .. code-block:: python
        del da[0]  # remove first element from DocumentArray
        del da[0:5]  # remove elements for 0 to 5 from DocumentArray
    """

    _data: MutableSequence[T_item]

    @abc.abstractmethod
    def __init__(
        self,
        docs: Optional[Iterable[T_item]] = None,
    ):
        ...

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
        torch_available = is_torch_available()
        if torch_available:
            import torch
        else:
            raise ValueError(f'Invalid index type {type(item)}')
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

    def _set_by_indices(self: T, item: Iterable[int], value: Iterable[T]):
        # here we cannot use _get_offset_to_doc() because we need to change the doc
        # that a given offset points to, not just retrieve it.
        # Future optimization idea: _data could be List[DocContainer], where
        # DocContainer points to the doc. Then we could use _get_offset_to_container()
        # to swap the doc in the container.
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

    def __delitem__(self, key: Union[int, IndexIterType]) -> None:
        key = self._normalize_index_item(key)

        if key is None:
            return

        del self._data[key]

    @overload
    def __getitem__(self: T, item: int) -> T:
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
    def __setitem__(self: T, key: IndexIterType, value: T):
        ...

    @overload
    def __setitem__(self: T, key: int, value: T):
        ...

    def __setitem__(self: T, key: Union[int, IndexIterType], value: Union[T, T_item]):
        key_norm = self._normalize_index_item(key)

        if isinstance(key_norm, int):
            value_int = cast(T_item, value)
            self._data[key_norm] = value_int
        elif isinstance(key_norm, slice):
            value_slice = cast(T, value)
            self._data[key_norm] = value_slice
        else:
            # _normalize_index_item() guarantees the line below is correct
            head = key_norm[0]  # type: ignore
            if isinstance(head, bool):
                key_norm_ = cast(Iterable[bool], key_norm)
                value_ = cast(Sequence[T_item], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_mask(key_norm_, value_)
            elif isinstance(head, int):
                key_norm__ = cast(Iterable[int], key_norm)
                value_ = cast(Sequence[T_item], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_indices(key_norm__, value_)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')
