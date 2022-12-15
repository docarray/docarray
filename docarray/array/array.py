from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Iterable, List, Type, TypeVar, Union

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.document import AnyDocument, BaseDocument

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import DocumentArrayProto
    from docarray.typing import NdArray, TorchTensor


T = TypeVar('T', bound='DocumentArray')


def _delegate_meth_to_data(meth_name: str) -> None:

    func = getattr(list, meth_name)

    @wraps(func)
    def _delegate_meth(self, *args, **kwargs):
        return getattr(self._data, meth_name)(*args, **kwargs)

    return _delegate_meth


class DocumentArray(AbstractDocumentArray):
    document_type: Type[BaseDocument] = AnyDocument

    def __init__(self, docs: Iterable[BaseDocument]):
        self._data = [doc_ for doc_ in docs]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
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

        if issubclass(field_type, BaseDocument):
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
            cls.document_type.from_protobuf(doc_proto)
            for doc_proto in pb_msg.list_.docs
        )

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import DocumentArrayListProto, DocumentArrayProto

        da_proto = DocumentArrayListProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        return DocumentArrayProto(list_=da_proto)

    @contextmanager
    def stacked_mode(self):
        """
        Context manager to put the DocumentArray in stacked mode and unstack it when
        exiting the context manager.
        EXAMPLE USAGE
        .. code-block:: python
            with da.stacked_mode():
                ...
        """

        from docarray.array.array_stacked import DocumentArrayStacked

        try:
            da_stacked = DocumentArrayStacked(self)
            yield da_stacked
        finally:
            self = DocumentArrayStacked.__class_getitem__(
                self.document_type
            ).to_document_array(da_stacked)

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
