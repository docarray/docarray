from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from typing_inspect import is_union_type

from docarray import BaseDocument, DocumentArray
from docarray.array.abstract_array import AnyDocumentArray
from docarray.typing import AnyTensor
from docarray.utils.find import FindResult
from docarray.utils.protocols import IsDataclass

TSchema = TypeVar('TSchema', bound=BaseDocument)


class BaseDocumentStore(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    # the BaseDocument that defines the schema of the store
    _schema: Type = type(None)  # this is filled automatically
    # columns with types according to the Document schema
    _columns_schema: Dict[str, Type] = {}  # this is filled automatically
    # default configurations for every column type
    # a dictionary from a column type (DB specific) to a dictionary
    # of default configurations for that type
    # These configs are used if no configs are specified in the `Field(...)`
    # of a field in the Document schema (`cls._schema`)
    # Example: `_default_column_config['VARCHAR'] = {'length': 255}`
    _default_column_config: Dict[Any, Dict[str, Any]] = {}

    def __init__(self, config: Optional[IsDataclass] = None):
        if issubclass(self._schema, type(None)):
            raise ValueError(
                'A DocumentStore must be typed with a Document type.'
                'To do so, use the syntax: DocumentStore[DocumentType]'
            )
        self._config = config
        # columns with DB specific types
        self._columns_db = {
            name: self.python_type_to_db_type(type_)
            for name, type_ in self._columns_schema.items()
        }
        self._column_configs: Dict[Any, Dict[str, Any]] = self._get_column_configs()

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
            _columns_schema: Dict[str, Type] = cls._unwrap_columns(_schema)

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

    def _is_schema_compatible(self, docs: Sequence[BaseDocument]) -> bool:
        """Flatten a DocumentArray into a DocumentArray of the schema type."""
        if isinstance(docs, AnyDocumentArray):
            docs_columns = self._unwrap_columns(docs.document_type)
            # this could be relaxed in the future,
            # see schema translation ideas in the design doc
            return docs_columns == self._columns_schema
        else:
            for d in docs:
                doc_columns = self._unwrap_columns(type(d))
                # this could be relaxed in the future,
                # see schema translation ideas in the design doc
                if doc_columns != self._columns_schema:
                    return False
            return True

    @staticmethod
    def get_value(doc: BaseDocument, col_name: str) -> Any:
        """Get the value of a column of a document."""
        if '__' in col_name:
            fields = col_name.split('__')
            leaf_doc: BaseDocument = doc
            for f in fields[:-1]:
                leaf_doc = getattr(leaf_doc, f)
            return getattr(leaf_doc, fields[-1])
        else:
            return getattr(doc, col_name)

    def get_data_by_columns(
        self, docs: Union[BaseDocument, Sequence[BaseDocument]]
    ) -> Dict[str, Generator[Any, None, None]]:
        """Get the payload of a document."""
        if isinstance(docs, BaseDocument):
            docs = [docs]
        if not self._is_schema_compatible(docs):
            raise ValueError(
                'The schema of the documents to be indexed is not compatible'
                ' with the schema of the store.'
            )
        return {
            col_name: (self.get_value(doc, col_name) for doc in docs)
            for col_name in self._columns_schema
        }

    def _get_column_configs(self) -> Dict[str, Dict[str, Any]]:
        col_configs: Dict[str, Dict[str, Any]] = {}
        for col_name, db_type in self._columns_db.items():
            col_configs[col_name] = self._default_column_config[db_type].copy()

            nestings = col_name.split('__')
            nested_fields, leaf_field = nestings[:-1], nestings[-1]
            leaf_doc: Type[BaseDocument] = self._schema
            for nested_field in nested_fields:
                # traverse nested document fields
                leaf_doc = leaf_doc.__fields__[nested_field].type_
            custom_config = leaf_doc.__fields__[leaf_field].field_info.extra
            col_configs[col_name].update(custom_config)

        return col_configs
