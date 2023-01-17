from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Type,
    TypeVar,
    Union,
    cast,
)

from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array import DocumentArray
from docarray.base_document import AnyDocument, BaseDocument
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._typing import is_tensor_union

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import DocumentArrayStackedProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor


try:
    from docarray.typing import TorchTensor
except ImportError:
    TorchTensor = None  # type: ignore

T = TypeVar('T', bound='DocumentArrayStacked')


class DocumentArrayStacked(AnyDocumentArray):
    """
    DocumentArrayStacked is a container of Documents appropriates to perform
    computation that require batches of data (ex: matrix multiplication, distance
    calculation, deep learning forward pass)

    A DocumentArrayStacked is similar to {class}`~docarray.array.DocumentArray`
    but the field of the Document that are {class}`~docarray.typing.AnyTensor` are
    stacked into a batches of AnyTensor. Like {class}`~docarray.array.DocumentArray`
    you can be precise a Document schema by using the `DocumentArray[MyDocument]`
    syntax where MyDocument is a Document class  (i.e. schema).
    This creates a DocumentArray that can only contains Documents of
    the type 'MyDocument'.

    :param docs: a DocumentArray

    """

    document_type: Type[BaseDocument] = AnyDocument
    _docs: DocumentArray

    def __init__(
        self: T,
        docs: DocumentArray,
    ):
        self._columns: Dict[str, Union['DocumentArrayStacked', AbstractTensor]] = {}

        self.from_document_array(docs)

    def from_document_array(self: T, docs: DocumentArray):
        self._docs = docs
        self.tensor_type = self._docs.tensor_type
        self._columns = self._create_columns(docs, tensor_type=self.tensor_type)

    @classmethod
    def _from_columns(
        cls: Type[T],
        docs: DocumentArray,
        columns: Mapping[str, Union['DocumentArrayStacked', AbstractTensor]],
    ) -> T:
        # below __class_getitem__ is called explicitly instead
        # of doing DocumentArrayStacked[docs.document_type]
        # because mypy has issues with class[...] notation at runtime.
        # see bug here: https://github.com/python/mypy/issues/13026
        # as of 2023-01-05 it should be fixed on mypy master, though, see
        # here: https://github.com/python/typeshed/issues/4819#issuecomment-1354506442
        da_stacked = DocumentArray.__class_getitem__(cls.document_type)([]).stack()
        da_stacked._columns = columns
        da_stacked._docs = docs
        return da_stacked

    def to(self: T, device: str) -> T:
        """Move all tensors of this DocumentArrayStacked to the given device

        :param device: the device to move the data to
        """
        for field in self._columns.keys():
            col = self._columns[field]
            if isinstance(col, AbstractTensor):
                self._columns[field] = col.__class__._docarray_from_native(
                    col.get_comp_backend().to_device(col, device)
                )
            else:  # recursive call
                col_docarray = cast(T, col)
                col_docarray.to(device)
        return self

    @classmethod
    def _get_columns_schema(
        cls: Type[T],
        tensor_type: Type[AbstractTensor],
    ) -> Mapping[str, Union[Type[AbstractTensor], Type[BaseDocument]]]:
        """
        Return the list of fields that are tensors and the list of fields that are
        documents
        :param tensor_type: the default tensor type fallback in case of union of tensor
        :return: a tuple of two lists, the first one is the list of fields that are
        tensors, the second one is the list of fields that are documents
        """

        column_schema: Dict[str, Union[Type[AbstractTensor], Type[BaseDocument]]] = {}

        for field_name, field in cls.document_type.__fields__.items():
            field_type = field.outer_type_
            if is_tensor_union(field_type):
                column_schema[field_name] = tensor_type
            elif isinstance(field_type, type):
                if issubclass(field_type, (BaseDocument, AbstractTensor)):
                    column_schema[field_name] = field_type

        return column_schema

    @classmethod
    def _create_columns(
        cls: Type[T], docs: DocumentArray, tensor_type: Type[AbstractTensor]
    ) -> Dict[str, Union['DocumentArrayStacked', AbstractTensor]]:

        if len(docs) == 0:
            return {}

        column_schema = cls._get_columns_schema(tensor_type)

        columns: Dict[str, Union[DocumentArrayStacked, AbstractTensor]] = dict()

        for field, type_ in column_schema.items():
            if issubclass(type_, AbstractTensor):
                tensor = getattr(docs[0], field)
                column_shape = (
                    (len(docs), *tensor.shape) if tensor is not None else (len(docs),)
                )
                columns[field] = type_._docarray_from_native(
                    type_.get_comp_backend().empty(column_shape)
                )

                for i, doc in enumerate(docs):
                    val = getattr(doc, field)
                    if val is None:
                        val = tensor_type.get_comp_backend().none_value()

                    cast(AbstractTensor, columns[field])[i] = val
                    setattr(doc, field, columns[field][i])
                    del val

            elif issubclass(type_, BaseDocument):
                columns[field] = getattr(docs, field).stack()

        return columns

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, 'DocumentArrayStacked', AbstractTensor]:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._columns.keys():
            return self._columns[field]
        else:
            return getattr(self._docs, field)

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, AbstractTensor],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        if field in self._columns.keys() and not isinstance(values, List):
            self._columns[field] = values
        else:
            setattr(self._docs, field, values)

    def __getitem__(self, item):  # note this should handle slices
        if isinstance(item, slice):
            return self._get_slice(item)
        doc = self._docs[item]
        # NOTE: this could be speed up by using a cache
        for field in self._columns.keys():
            setattr(doc, field, self._columns[field][item])
        return doc

    def _get_slice(self: T, item: slice) -> T:
        """Return a slice of the DocumentArrayStacked

        :param item: the slice to apply
        :return: a DocumentArrayStacked
        """

        columns_sliced = {k: col[item] for k, col in self._columns.items()}
        columns_sliced_ = cast(Dict[str, Union[AbstractTensor, T]], columns_sliced)
        return self._from_columns(self._docs[item], columns_sliced_)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._docs)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayStackedProto') -> T:
        """create a Document from a protobuf message"""

        docs = DocumentArray(
            cls.document_type.from_protobuf(doc_proto)
            for doc_proto in pb_msg.list_.docs
        )
        da = cls(DocumentArray([]))

        da._docs = docs
        da._columns = pb_msg.columns
        return da

    def to_protobuf(self) -> 'DocumentArrayStackedProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import (
            DocumentArrayProto,
            DocumentArrayStackedProto,
            UnionArrayProto,
        )

        da_proto = DocumentArrayProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        columns_proto: Dict[str, UnionArrayProto] = dict()
        for field, column in self._columns.items():
            if isinstance(column, DocumentArrayStacked):
                columns_proto[field] = UnionArrayProto(
                    document_array=DocumentArrayProto(stack=column.to_protobuf())
                )
            elif isinstance(column, AbstractTensor):
                columns_proto[field] = UnionArrayProto(ndarray=column.to_protobuf())

        return DocumentArrayStackedProto(list_=da_proto, columns=columns_proto)

    def unstack(self: T) -> DocumentArray:
        """Convert DocumentArrayStacked into a DocumentArray.

        Note this destroys the arguments and returns a new DocumentArray
        """
        for i, doc in enumerate(self._docs):
            for field in self._columns.keys():
                val = self._columns[field]
                setattr(doc, field, val[i])

                # NOTE: here we might need to copy the tensor
                # see here
                # https://discuss.pytorch.org/t/what-happened-to-a-view-of-a-tensor
                # -when-the-original-tensor-is-deleted/167294 # noqa: E501

        for field in list(self._columns.keys()):
            # list needed here otherwise we are modifying the dict while iterating
            del self._columns[field]

        da_list = self._docs
        return da_list

    @contextmanager
    def unstacked_mode(self):
        """
        Context manager to put the DocumentArrayStacked in unstacked mode and stack it
        when exiting the context manager.
        EXAMPLE USAGE
        .. code-block:: python
            with da.unstacked_mode():
                ...
        """
        try:
            da_unstacked = DocumentArrayStacked.unstack(self)
            yield da_unstacked
        finally:
            self._docs = da_unstacked
            self.from_document_array(da_unstacked)

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
            return cls(DocumentArray(value))
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def traverse_flat(
        self: 'AnyDocumentArray',
        access_path: str,
    ) -> Union[List[Any], 'TorchTensor', 'NdArray']:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        cls_to_check = (NdArray, TorchTensor) if TorchTensor is not None else (NdArray,)

        if len(flattened) == 1 and isinstance(flattened[0], cls_to_check):
            return flattened[0]
        else:
            return flattened
