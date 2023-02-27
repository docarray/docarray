import io
from contextlib import contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from typing_inspect import is_union_type

from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array.io import IOMixinArray
from docarray.base_document import AnyDocument, BaseDocument
from docarray.typing import NdArray
from docarray.utils.misc import is_torch_available

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.array.stacked.array_stacked import DocumentArrayStacked
    from docarray.proto import DocumentArrayProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='DocumentArray')
T_doc = TypeVar('T_doc', bound=BaseDocument)
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


def _delegate_meth_to_data(meth_name: str) -> Callable:
    """
    create a function that mimic a function call to the data attribute of the
    DocumentArray

    :param meth_name: name of the method
    :return: a method that mimic the meth_name
    """
    func = getattr(list, meth_name)

    @wraps(func)
    def _delegate_meth(self, *args, **kwargs):
        return getattr(self._data, meth_name)(*args, **kwargs)

    return _delegate_meth


def _is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it


class DocumentArray(IOMixinArray, AnyDocumentArray[T_doc]):
    """
     DocumentArray is a container of Documents.

    :param docs: iterable of Document
    :param tensor_type: Class used to wrap the tensors of the Documents when stacked

    A DocumentArray is a list of Documents of any schema. However, many
    DocumentArray features are only available if these Documents are
    homogeneous and follow the same schema. To precise this schema you can use
    the `DocumentArray[MyDocument]` syntax where MyDocument is a Document class
    (i.e. schema). This creates a DocumentArray that can only contains Documents of
    the type 'MyDocument'.

    EXAMPLE USAGE
    .. code-block:: python
        from docarray import BaseDocument, DocumentArray
        from docarray.typing import NdArray, ImageUrl
        from typing import Optional


        class Image(BaseDocument):
            tensor: Optional[NdArray[100]]
            url: ImageUrl


        da = DocumentArray[Image](
            Image(url='http://url.com/foo.png') for _ in range(10)
        )  # noqa: E510


    If your DocumentArray is homogeneous (i.e. follows the same schema), you can access
    fields at the DocumentArray level (for example `da.tensor` or `da.url`).
    You can also set fields, with `da.tensor = np.random.random([10, 100])`:


    .. code-block:: python
        print(da.url)
        # [ImageUrl('http://url.com/foo.png', host_type='domain'), ...]
        import numpy as np

        da.tensor = np.random.random([10, 100])
        print(da.tensor)
        # [NdArray([0.11299577, 0.47206767, 0.481723  , 0.34754724, 0.15016037,
        #          0.88861321, 0.88317666, 0.93845579, 0.60486676, ... ]), ...]

    You can index into a DocumentArray like a numpy array or torch tensor:

    .. code-block:: python
        da[0]  # index by position
        da[0:5:2]  # index by slice
        da[[0, 2, 3]]  # index by list of indices
        da[True, False, True, True, ...]  # index by boolean mask

    You can delete items from a DocumentArray like a Python List

    .. code-block:: python
        del da[0]  # remove first element from DocumentArray
        del da[0:5]  # remove elements fro 0 to 5 from DocumentArray

    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(
        self,
        docs: Optional[Iterable[T_doc]] = None,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):

        self._data: List[T_doc] = list(self._validate_docs(docs)) if docs else []
        self.tensor_type = tensor_type

    def _validate_docs(self, docs: Iterable[T_doc]) -> Iterable[T_doc]:
        """
        Validate if an Iterable of Document are compatible with this DocumentArray
        """
        for doc in docs:
            yield self._validate_one_doc(doc)

    def _validate_one_doc(self, doc: T_doc) -> T_doc:
        """Validate if a Document is compatible with this DocumentArray"""
        if not issubclass(self.document_type, AnyDocument) and not isinstance(
            doc, self.document_type
        ):
            raise ValueError(f'{doc} is not a {self.document_type}')
        return doc

    def __len__(self):
        return len(self._data)

    @overload
    def __getitem__(self: T, item: int) -> BaseDocument:
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
    def __setitem__(self: T, key: int, value: T_doc):
        ...

    def __setitem__(self: T, key: Union[int, IndexIterType], value: Union[T, T_doc]):
        key_norm = self._normalize_index_item(key)

        if isinstance(key_norm, int):
            value_int = cast(BaseDocument, value)
            self._data[key_norm] = value_int
        elif isinstance(key_norm, slice):
            value_slice = cast(T, value)
            self._data[key_norm] = value_slice
        else:
            # _normalize_index_item() guarantees the line below is correct
            head = key_norm[0]  # type: ignore
            if isinstance(head, bool):
                key_norm_ = cast(Iterable[bool], key_norm)
                value_ = cast(Sequence[BaseDocument], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_mask(key_norm_, value_)
            elif isinstance(head, int):
                key_norm__ = cast(Iterable[int], key_norm)
                value_ = cast(Sequence[BaseDocument], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_indices(key_norm__, value_)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')

    def __iter__(self):
        return iter(self._data)

    @overload
    def __delitem__(self: T, key: int) -> None:
        ...

    @overload
    def __delitem__(self: T, key: IndexIterType) -> None:
        ...

    def __delitem__(self, key) -> None:
        key = self._normalize_index_item(key)

        if key is None:
            return

        del self._data[key]

    def __bytes__(self) -> bytes:
        with io.BytesIO() as bf:
            self._write_bytes(bf=bf)
            return bf.getvalue()

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

    def _set_by_indices(self: T, item: Iterable[int], value: Iterable[BaseDocument]):
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
            (doc for doc, mask_value in zip(self, item) if mask_value)
        )

    def _set_by_mask(self: T, item: Iterable[bool], value: Sequence[BaseDocument]):
        i_value = 0
        for i, mask_value in zip(range(len(self)), item):
            if mask_value:
                self._data[i] = value[i_value]
                i_value += 1

    def append(self, doc: T_doc):
        """
        Append a Document to the DocumentArray. The Document must be from the same class
        as the document_type of this DocumentArray otherwise it will fail.
        :param doc: A Document
        """
        self._data.append(self._validate_one_doc(doc))

    def extend(self, docs: Iterable[T_doc]):
        """
        Extend a DocumentArray with an Iterable of Document. The Documents must be from
        the same class as the document_type of this DocumentArray otherwise it will
        fail.
        :param docs: Iterable of Documents
        """
        self._data.extend(self._validate_docs(docs))

    def insert(self, i: int, doc: T_doc):
        """
        Insert a Document to the DocumentArray. The Document must be from the same
        class as the document_type of this DocumentArray otherwise it will fail.
        :param i: index to insert
        :param doc: A Document
        """
        self._data.insert(i, self._validate_one_doc(doc))

    pop = _delegate_meth_to_data('pop')
    remove = _delegate_meth_to_data('remove')
    reverse = _delegate_meth_to_data('reverse')
    sort = _delegate_meth_to_data('sort')

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        field_type = self.__class__.document_type._get_field_type(field)

        if (
            not is_union_type(field_type)
            and isinstance(field_type, type)
            and issubclass(field_type, BaseDocument)
        ):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return DocumentArray.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self), tensor_type=self.tensor_type
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, 'AbstractTensor'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        ...

        for doc, value in zip(self, values):
            setattr(doc, field, value)

    @contextmanager
    def stacked_mode(self):
        """
        Context manager to convert DocumentArray to a DocumentArrayStacked and unstack
        it when exiting the context manager.
        EXAMPLE USAGE
        .. code-block:: python
            with da.stacked_mode():
                ...
        """

        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        try:
            da_stacked = DocumentArrayStacked.__class_getitem__(self.document_type)(
                self,
            )
            yield da_stacked
        finally:
            self = DocumentArrayStacked.__class_getitem__(self.document_type).unstack(
                da_stacked
            )

    def stack(self) -> 'DocumentArrayStacked':
        """
        Convert the DocumentArray into a DocumentArrayStacked. `Self` cannot be used
        afterwards
        """
        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        return DocumentArrayStacked.__class_getitem__(self.document_type)(self)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[BaseDocument]],
        field: 'ModelField',
        config: 'BaseConfig',
    ):
        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        if isinstance(value, (cls, DocumentArrayStacked)):
            return value
        elif isinstance(value, Iterable):
            return cls(value)
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def traverse_flat(
        self: 'DocumentArray',
        access_path: str,
    ) -> List[Any]:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        return flattened

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message
        :param pb_msg: The protobuf message from where to construct the DocumentArray
        """
        return super().from_protobuf(pb_msg)
