from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Type, TypeVar, Union

from typing_inspect import is_union_type

from docarray.array.abstract_array import AnyDocumentArray
from docarray.document import AnyDocument, BaseDocument
from docarray.typing import NdArray

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.array.array_stacked import DocumentArrayStacked
    from docarray.proto import DocumentArrayProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor


T = TypeVar('T', bound='DocumentArray')


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


class DocumentArray(AnyDocumentArray):
    """
     DocumentArray is a container of Documents.

    :param docs: iterable of Document

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


        class Image(BaseDocument):
            tensor: Optional[NdArray[100]]
            url: ImageUrl


        da = DocumentArray[Image](
            Image(url='http://url.com/foo.png') for _ in range(10)
        )  # noqa: E510


    If your DocumentArray is homogeneous (i.e. follows the same schema), you can access
    fields at the DocumentArray level (for example `da.tensor`). You can also set
    fields, with `da.tensor = np.random.random([10, 100])`
    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(
        self,
        docs: Iterable[BaseDocument],
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):
        self._data = [doc_ for doc_ in docs]
        self.tensor_type = tensor_type

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if type(item) == slice:
            return self.__class__(self._data[item])
        else:
            return self._data[item]

    def __iter__(self):
        return iter(self._data)

    append = _delegate_meth_to_data('append')
    extend = _delegate_meth_to_data('extend')
    insert = _delegate_meth_to_data('insert')
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
        field_type = self.__class__.document_type._get_nested_document_class(field)

        if not is_union_type(field_type) and issubclass(field_type, BaseDocument):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return self.__class__.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self)
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, 'TorchTensor', 'NdArray'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        ...

        for doc, value in zip(self, values):
            setattr(doc, field, value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message"""
        return cls(
            cls.document_type.from_protobuf(doc_proto) for doc_proto in pb_msg.docs
        )

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import DocumentArrayProto

        da_proto = DocumentArrayProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        return da_proto

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

        from docarray.array.array_stacked import DocumentArrayStacked

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
        from docarray.array.array_stacked import DocumentArrayStacked

        return DocumentArrayStacked.__class_getitem__(self.document_type)(self)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[BaseDocument]],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        elif isinstance(value, Iterable):
            return cls(value)
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def traverse_flat(
        self: 'DocumentArray',
        access_path: str,
    ) -> Union[List[Any]]:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        return flattened
