from typing import (
    Any,
    Iterable,
    List,
    Sequence,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

import numpy as np
from typing_extensions import SupportsIndex

from docarray.utils._internal.misc import (
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

torch_available = is_torch_available()
if torch_available:
    import torch
tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor
jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

    from docarray.typing.tensor.jaxarray import JaxArray

T_item = TypeVar('T_item')
T = TypeVar('T', bound='ListAdvancedIndexing')

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


class ListAdvancedIndexing(List[T_item]):
    """
    A list wrapper that implements custom indexing

    You can index into a ListAdvanceIndex like a numpy array or torch tensor:

    ---

    ```python
    docs[0]  # index by position
    docs[0:5:2]  # index by slice
    docs[[0, 2, 3]]  # index by list of indices
    docs[True, False, True, True, ...]  # index by boolean mask
    ```

    ---

    """

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
        if torch_available:

            allowed_torch_dtypes = [
                torch.bool,
                torch.int64,
            ]
            if isinstance(item, torch.Tensor) and (item.dtype in allowed_torch_dtypes):
                return item.tolist()

        if tf_available:
            if isinstance(item, tf.Tensor):
                return item.numpy().tolist()
            if isinstance(item, TensorFlowTensor):
                return item.tensor.numpy().tolist()

        if jax_available:
            if isinstance(item, jnp.ndarray):
                return item.__array__().tolist()
            if isinstance(item, JaxArray):
                return item.tensor.__array__().tolist()

        return item

    def _get_from_indices(self: T, item: Iterable[int]) -> T:
        results = []
        for ix in item:
            results.append(self[ix])
        return self.__class__(results)

    def _set_by_indices(self: T, item: Iterable[int], value: Iterable[T_item]):
        for ix, doc_to_set in zip(item, value):
            try:
                self[ix] = doc_to_set
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
                self[i] = value[i_value]
                i_value += 1

    def _del_from_mask(self: T, item: Iterable[bool]) -> None:
        idx_to_delete = [i for i, val in enumerate(item) if val]
        self._del_from_indices(idx_to_delete)

    def _del_from_indices(self: T, item: Iterable[int]) -> None:
        for ix in sorted(item, reverse=True):
            # reversed is needed here otherwise some the indices are not up to date after
            # each delete
            del self[ix]

    def __delitem__(self, key: Union[SupportsIndex, IndexIterType]) -> None:
        item = self._normalize_index_item(key)

        if item is None:
            return
        elif isinstance(item, (int, slice)):
            super().__delitem__(item)
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
    def __getitem__(self: T, item: SupportsIndex) -> T_item:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item):
        item = self._normalize_index_item(item)

        if type(item) == slice:
            return self.__class__(super().__getitem__(item))

        if isinstance(item, int):
            return super().__getitem__(item)

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
    def __setitem__(self: T, key: SupportsIndex, value: T_item) -> None:
        ...

    @overload
    def __setitem__(self: T, key: IndexIterType, value: Iterable[T_item]):
        ...

    @no_type_check
    def __setitem__(self: T, key, value):
        key_norm = self._normalize_index_item(key)

        if isinstance(key_norm, int):
            super().__setitem__(key_norm, value)
        elif isinstance(key_norm, slice):
            super().__setitem__(key_norm, value)
        else:
            # _normalize_index_item() guarantees the line below is correct
            head = key_norm[0]
            if isinstance(head, bool):
                return self._set_by_mask(key_norm, value)
            elif isinstance(head, int):
                return self._set_by_indices(key_norm, value)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')
