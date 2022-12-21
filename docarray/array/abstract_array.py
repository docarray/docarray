from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Sequence, Type, TypeVar, Union

from docarray.document import BaseDocument
from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from docarray import Document
    from docarray.proto import DocumentArrayProto, NodeProto
    from docarray.typing import NdArray, Tensor, TorchTensor

T = TypeVar('T', bound='AnyDocumentArray')
T_doc = TypeVar('T_doc', bound=BaseDocument)


class AnyDocumentArray(Sequence[BaseDocument], Generic[T_doc], AbstractType):
    document_type: Type[BaseDocument]

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        class _DocumentArrayTyped(cls):  # type: ignore
            document_type: Type[BaseDocument] = item

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

        _DocumentArrayTyped.__name__ = f'DocumentArray[{item.__name__}]'
        _DocumentArrayTyped.__qualname__ = f'DocumentArray[{item.__name__}]'

        return _DocumentArrayTyped

    @abstractmethod
    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
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
        values: Union[List, T, 'TorchTensor', 'NdArray'],
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

        return NodeProto(chunks=self.to_protobuf())

    def traverse_flat(
        self: 'AnyDocumentArray',
        access_path: str,
    ) -> Union[List[Any], 'Tensor']:
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
            from docarray import Document, DocumentArray, Text


            class Author(Document):
                name: str


            class Book(Document):
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
            from docarray import Document, DocumentArray


            class Chapter(Document):
                content: str


            class Book(Document):
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

        If your DocumentArray is in stacked mode and you want to access a field of
        type Tensor, the stacked tensor will be returned instead of a list:

        EXAMPLE USAGE
        .. code-block:: python
            class Image(Document):
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
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten(nodes)

        from docarray.typing import Tensor

        if len(flattened) == 1 and isinstance(flattened[0], Tensor):
            return flattened[0]
        else:
            return flattened

    @staticmethod
    def _traverse(node: Union['Document', 'AnyDocumentArray'], access_path: str):
        if access_path:
            path_attrs = access_path.split('.')
            curr_attr = path_attrs[0]
            path_attrs.pop(0)

            from docarray.array import DocumentArrayStacked

            if isinstance(node, (AnyDocumentArray, list)) and not isinstance(
                node, DocumentArrayStacked
            ):
                for n in node:
                    x = getattr(n, curr_attr)
                    yield from AnyDocumentArray._traverse(x, '.'.join(path_attrs))
            else:
                x = getattr(node, curr_attr)
                yield from AnyDocumentArray._traverse(x, '.'.join(path_attrs))
        else:
            yield node

    @staticmethod
    def _flatten(sequence: List[Any]) -> List[Any]:
        from docarray import DocumentArray

        res: List[Any] = []
        for seq in sequence:
            if isinstance(seq, (list, DocumentArray)):
                res += seq
            else:
                res.append(seq)

        return res
