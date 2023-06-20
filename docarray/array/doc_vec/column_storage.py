from collections import ChainMap
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    ItemsView,
    Iterable,
    MutableMapping,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    ValuesView,
)

from docarray.array.list_advance_indexing import ListAdvancedIndexing
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.array.doc_vec.doc_vec import DocVec

IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


T = TypeVar('T', bound='ColumnStorage')


class ColumnsJsonCompatible(NamedTuple):
    tensor_columns: Dict[str, Any]
    doc_columns: Dict[str, Any]
    docs_vec_columns: Dict[str, Any]
    any_columns: Dict[str, Any]


class ColumnStorage:
    """
    ColumnStorage is a container to store the columns of the
    :class:`~docarray.array.doc_vec.DocVec`.

    :param tensor_columns: a Dict of AbstractTensor
    :param doc_columns: a Dict of :class:`~docarray.array.doc_vec.DocVec`
    :param docs_vec_columns: a Dict of List of :class:`~docarray.array.doc_vec.DocVec`
    :param any_columns: a Dict of List
    :param tensor_type: Class used to wrap the doc_vec tensors
    """

    def __init__(
        self,
        tensor_columns: Dict[str, Optional[AbstractTensor]],
        doc_columns: Dict[str, Optional['DocVec']],
        docs_vec_columns: Dict[str, Optional[ListAdvancedIndexing['DocVec']]],
        any_columns: Dict[str, ListAdvancedIndexing],
        tensor_type: Type[AbstractTensor] = NdArray,
    ):
        self.tensor_columns = tensor_columns
        self.doc_columns = doc_columns
        self.docs_vec_columns = docs_vec_columns
        self.any_columns = any_columns

        self.tensor_type = tensor_type

        self.columns = ChainMap(  # type: ignore
            self.tensor_columns,  # type: ignore
            self.doc_columns,  # type: ignore
            self.docs_vec_columns,  # type: ignore
            self.any_columns,  # type: ignore
        )  # type: ignore

    def __len__(self) -> int:
        return len(self.any_columns['id'])  # TODO what if ID are None ?

    def __getitem__(self: T, item: IndexIterType) -> T:
        if isinstance(item, tuple):
            item = list(item)
        tensor_columns = {
            key: col[item] if col is not None else None
            for key, col in self.tensor_columns.items()
        }
        doc_columns = {
            key: col[item] if col is not None else None
            for key, col in self.doc_columns.items()
        }
        docs_vec_columns = {
            key: col[item] if col is not None else None
            for key, col in self.docs_vec_columns.items()
        }
        any_columns = {
            key: col[item] if col is not None else None
            for key, col in self.any_columns.items()
        }

        return self.__class__(
            tensor_columns,
            doc_columns,
            docs_vec_columns,
            any_columns,
            self.tensor_type,
        )

    def columns_json_compatible(self) -> ColumnsJsonCompatible:
        tens_cols = {
            key: value._docarray_to_json_compatible() if value is not None else value
            for key, value in self.tensor_columns.items()
        }
        doc_cols = {
            key: value._docarray_to_json_compatible() if value is not None else value
            for key, value in self.doc_columns.items()
        }
        doc_vec_cols = {
            key: [vec._docarray_to_json_compatible() for vec in value]
            if value is not None
            else value
            for key, value in self.docs_vec_columns.items()
        }
        return ColumnsJsonCompatible(
            tens_cols, doc_cols, doc_vec_cols, self.any_columns
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ColumnStorage):
            return False
        if self.tensor_type != other.tensor_type:
            return False
        for col_map_self, col_map_other in zip(self.columns.maps, other.columns.maps):
            if col_map_self.keys() != col_map_other.keys():
                return False
            for key_self in col_map_self.keys():
                if key_self == 'id':
                    continue

                val1, val2 = col_map_self[key_self], col_map_other[key_self]
                if isinstance(val1, AbstractTensor):
                    values_are_equal = val1.get_comp_backend().equal(val1, val2)
                elif isinstance(val2, AbstractTensor):
                    values_are_equal = val2.get_comp_backend().equal(val1, val2)
                else:
                    values_are_equal = val1 == val2
                if not values_are_equal:
                    return False
        return True


class ColumnStorageView(dict, MutableMapping[str, Any]):
    index: int
    storage: ColumnStorage

    def __init__(self, index: int, storage: ColumnStorage):
        super().__init__()
        self.index = index
        self.storage = storage

    def __getitem__(self, name: str) -> Any:
        if name in self.storage.tensor_columns.keys():
            tensor = self.storage.tensor_columns[name]
            if tensor is None:
                return None
            if tensor.get_comp_backend().n_dim(tensor) == 1:
                # to ensure consistensy between numpy and pytorch
                # we wrap the scalr in a tensor of ndim = 1
                # otherwise numpy pass by value whereas torch by reference
                col = self.storage.tensor_columns[name]

                if col is not None:
                    return col[self.index : self.index + 1]
                else:
                    return None

        col = self.storage.columns[name]

        if col is None:
            return None
        return col[self.index]

    def __reduce__(self):
        # implementing __reduce__ to solve a pickle issue when subclassing dict
        # see here: https://stackoverflow.com/questions/21144845/how-can-i-unpickle-a-subclass-of-dict-that-validates-with-setitem-in-pytho
        return (ColumnStorageView, (self.index, self.storage))

    def __setitem__(self, name, value) -> None:
        if self.storage.columns[name] is None:
            raise ValueError(
                f'Cannot set an item to a None column. This mean that '
                f'the DocVec that encapsulate this doc has the field '
                f'{name} set to None. If you want to modify that you need to do it at the'
                f'DocVec level. `docs.field = np.zeros(10)`'
            )

        self.storage.columns[name][self.index] = value

    def __delitem__(self, key):
        raise RuntimeError('Cannot delete an item from a StorageView')

    def __iter__(self):
        return self.storage.columns.keys()

    def __len__(self):
        return len(self.storage.columns)

    def _local_dict(self):
        """The storage.columns dictionary with every value at position self.index"""

        return {key: self[key] for key in self.storage.columns.keys()}

    def keys(self):
        return self.storage.columns.keys()

    # type ignore because return type dict_values is private and we cannot use it.
    # context: https://github.com/python/typing/discussions/1033
    def values(self) -> ValuesView:  # type: ignore
        return ValuesView(self._local_dict())

    # type ignore because return type dict_items is private and we cannot use it.
    # context: https://github.com/python/typing/discussions/1033
    def items(self) -> ItemsView:  # type: ignore
        return ItemsView(self._local_dict())
