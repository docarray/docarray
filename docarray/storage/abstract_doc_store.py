from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar, Union

from typing_inspect import is_union_type

from docarray import BaseDocument, DocumentArray
from docarray.array.abstract_array import AnyDocumentArray
from docarray.typing import AnyTensor
from docarray.utils.find import FindResult

TSchema = TypeVar('TSchema', bound=BaseDocument)


class BaseDocumentStore(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    _schema: Type = type(None)
    _columns: Dict[str, Type] = {}

    def __init__(self):
        if issubclass(self._schema, type(None)):
            raise ValueError(
                'A DocumentStore must be typed with a Document type.'
                'To do so, use the syntax: DocumentStore[DocumentType]'
            )
        self._db_columns = {
            name: self.python_type_to_db_type(type_)
            for name, type_ in self._columns.items()
        }

    @abstractmethod
    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        ...

    @abstractmethod
    def index(self, docs: Union[TSchema, Sequence[TSchema]]):
        """Index a document into the store"""
        ...

    @abstractmethod
    def find(
        self,
        query: Union[AnyTensor, BaseDocument],
        embedding_field: str = 'embedding',
        metric: str = 'cosine_sim',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the store"""
        # TODO(johannes) refine method signature
        ...

    @abstractmethod
    def find_batched(
        self,
        query: Union[AnyTensor, DocumentArray],
        embedding_field: str = 'embedding',
        metric: str = 'cosine_sim',
        limit: int = 10,
        **kwargs,
    ) -> List[FindResult]:
        """Find documents in the store"""
        # TODO(johannes) refine method signature
        ...

    def __class_getitem__(cls, item: Type[TSchema]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        class _DocumentStoreTyped(cls):  # type: ignore
            _schema: Type[TSchema] = item
            _columns: Dict[str, Type] = cls._unwrap_columns(_schema)

        _DocumentStoreTyped.__name__ = f'{cls.__name__}[{item.__name__}]'
        _DocumentStoreTyped.__qualname__ = f'{cls.__qualname__}[{item.__name__}]'

        return _DocumentStoreTyped

    @classmethod
    def _unwrap_columns(cls, schema: Type[BaseDocument]) -> Dict[str, Type]:
        columns: Dict[str, Type] = dict()
        for field_name, field in schema.__fields__.items():
            t_ = field.type_
            if is_union_type(t_):
                # TODO(johannes): this restriction has to
                # go othws we can't even index built in docs
                raise ValueError(
                    'Indexing field of Union type is not'
                    f'supported. Instead of using type'
                    f'{t_} use a single specific type.'
                )
            elif issubclass(t_, AnyDocumentArray):
                raise ValueError(
                    'Indexing field of DocumentArray type (=subindex)'
                    'is not yet supported.'
                )
            elif issubclass(t_, BaseDocument):
                columns = dict(
                    columns,
                    **{
                        f'{field_name}__{nested_name}': t
                        for nested_name, t in cls._unwrap_columns(t_).items()
                    },
                )
            else:
                columns[field_name] = t_
        return columns
