from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from docarray.base_document import BaseDocument
from docarray.display.document_array_summary import DocumentArraySummary
from docarray.typing import NdArray
from docarray.typing.abstract_type import AbstractType
from docarray.utils._typing import change_cls_name

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto, NodeProto
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AnyDocumentArray')
T_doc = TypeVar('T_doc', bound=BaseDocument)


class AnyDocumentArray(Sequence[BaseDocument], Generic[T_doc], AbstractType):
    document_type: Type[BaseDocument]
    tensor_type: Type['AbstractTensor'] = NdArray
    __typed_da__: Dict[Type['AnyDocumentArray'], Dict[Type[BaseDocument], Type]] = {}

    def __repr__(self):
        return f'<{self.__class__.__name__} (length={len(self)})>'

    @classmethod
    def __class_getitem__(cls, item: Union[Type[BaseDocument], TypeVar, str]):
        if not isinstance(item, type):
            return Generic.__class_getitem__.__func__(cls, item)  # type: ignore
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        if cls not in cls.__typed_da__:
            cls.__typed_da__[cls] = {}

        if item not in cls.__typed_da__[cls]:
            # Promote to global scope so multiprocessing can pickle it
            global _DocumentArrayTyped

            class _DocumentArrayTyped(cls):  # type: ignore
                document_type: Type[BaseDocument] = cast(Type[BaseDocument], item)

            for field in _DocumentArrayTyped.document_type.__fields__.keys():

                def _property_generator(val: str):
                    def _getter(self):
                        return self._get_array_attribute(val)

                    def _setter(self, value):
                        self._set_array_attribute(val, value)

                    # need docstring for the property
                    return property(fget=_getter, fset=_setter)

                setattr(_DocumentArrayTyped, field, _property_generator(field))
                # this generates property on the fly based on the schema of the item

            # The global scope and qualname need to refer to this class a unique name.
            # Otherwise, creating another _DocumentArrayTyped will overwrite this one.
            change_cls_name(
                _DocumentArrayTyped, f'{cls.__name__}[{item.__name__}]', globals()
            )

            cls.__typed_da__[cls][item] = _DocumentArrayTyped

        return cls.__typed_da__[cls][item]

    @abstractmethod
    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'AbstractTensor']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        ...

    @abstractmethod
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

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message"""
        ...

    @abstractmethod
    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        ...

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert a DocumentArray into a NodeProto protobuf message.
         This function should be called when a DocumentArray
        is nested into another Document that need to be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(document_array=self.to_protobuf())

    @abstractmethod
    def traverse_flat(
        self: 'AnyDocumentArray',
        access_path: str,
    ) -> Union[List[Any], 'AbstractTensor']:
        """
        Return a List of the accessed objects when applying the access_path. If this
        results in a nested list or list of DocumentArrays, the list will be flattened
        on the first level. The access path is a string that consists of attribute
        names, concatenated and dot-seperated. It describes the path from the first
        level to an arbitrary one, e.g. 'doc_attr_x.sub_doc_attr_x.sub_sub_doc_attr_z'.

        :param access_path: a string that represents the access path.
        :return: list of the accessed objects, flattened if nested.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import BaseDocument, DocumentArray, Text


            class Author(BaseDocument):
                name: str


            class Book(BaseDocument):
                author: Author
                content: Text


            da = DocumentArray[Book](
                Book(author=Author(name='Jenny'), content=Text(text=f'book_{i}'))
                for i in range(10)  # noqa: E501
            )

            books = da.traverse_flat(access_path='content')  # list of 10 Text objs

            authors = da.traverse_flat(access_path='author.name')  # list of 10 strings

        If the resulting list is a nested list, it will be flattened:

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import BaseDocument, DocumentArray


            class Chapter(BaseDocument):
                content: str


            class Book(BaseDocument):
                chapters: DocumentArray[Chapter]


            da = DocumentArray[Book](
                Book(
                    chapters=DocumentArray[Chapter](
                        [Chapter(content='some_content') for _ in range(3)]
                    )
                )
                for _ in range(10)
            )

            chapters = da.traverse_flat(access_path='chapters')  # list of 30 strings

        If your DocumentArray is in stacked mode, and you want to access a field of
        type AnyTensor, the stacked tensor will be returned instead of a list:

        EXAMPLE USAGE
        .. code-block:: python
            class Image(BaseDocument):
                tensor: TorchTensor[3, 224, 224]


            batch = DocumentArray[Image](
                [
                    Image(
                        tensor=torch.zeros(3, 224, 224),
                    )
                    for _ in range(2)
                ]
            )

            batch_stacked = batch.stack()
            tensors = batch_stacked.traverse_flat(
                access_path='tensor'
            )  # tensor of shape (2, 3, 224, 224)

        """
        ...

    @staticmethod
    def _traverse(node: Any, access_path: str):
        if access_path:
            curr_attr, _, path_attrs = access_path.partition('.')

            from docarray.array import DocumentArray

            if isinstance(node, (DocumentArray, list)):
                for n in node:
                    x = getattr(n, curr_attr)
                    yield from AnyDocumentArray._traverse(x, path_attrs)
            else:
                x = getattr(node, curr_attr)
                yield from AnyDocumentArray._traverse(x, path_attrs)
        else:
            yield node

    @staticmethod
    def _flatten_one_level(sequence: List[Any]) -> List[Any]:
        from docarray import DocumentArray

        if len(sequence) == 0 or not isinstance(sequence[0], (list, DocumentArray)):
            return sequence
        else:
            return [item for sublist in sequence for item in sublist]

    def summary(self):
        """
        Print a summary of this DocumentArray object and a summary of the schema of its
        Document type.
        """
        DocumentArraySummary(self).summary()
