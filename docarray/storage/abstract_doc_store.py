from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from typing_inspect import is_union_type

from docarray import BaseDocument, DocumentArray
from docarray.array.abstract_array import AnyDocumentArray
from docarray.typing import AnyTensor
from docarray.utils.find import FindResult
from docarray.utils.protocols import IsDataclass

TSchema = TypeVar('TSchema', bound=BaseDocument)


class FindResultBatched(NamedTuple):
    documents: List[DocumentArray]
    scores: np.ndarray


@dataclass
class _Column:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]


class BaseDocumentStore(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    # the BaseDocument that defines the schema of the store
    # for subclasses this is filled automatically
    _schema: Optional[Type[BaseDocument]] = None

    # default configurations for every column type
    # a dictionary from a column type (DB specific) to a dictionary
    # of default configurations for that type
    # These configs are used if no configs are specified in the `Field(...)`
    # of a field in the Document schema (`cls._schema`)
    # Example: `_default_column_config['VARCHAR'] = {'length': 255}`
    _default_column_config: Dict[Any, Dict[str, Any]] = {}

    def __init__(self, config: Optional[IsDataclass] = None):
        if self._schema is None:
            raise ValueError(
                'A DocumentStore must be typed with a Document type.'
                'To do so, use the syntax: DocumentStore[DocumentType]'
            )
        self._config = config

        self._columns: Dict[str, _Column] = self._create_columns(self._schema)

    @abstractmethod
    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        ...

    @abstractmethod
    def index(self, docs: Union[TSchema, Sequence[TSchema]]):
        """Index a document into the store"""
        ...

    @abstractmethod
    def __delitem__(self, key: Union[str, Sequence[str]]):
        """Delete a document from the store, by `id`.

        :param key: id or ids to delete from the Document Store
        """
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
    ) -> FindResultBatched:
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

        _DocumentStoreTyped.__name__ = f'{cls.__name__}[{item.__name__}]'
        _DocumentStoreTyped.__qualname__ = f'{cls.__qualname__}[{item.__name__}]'

        return _DocumentStoreTyped

    def _create_columns(self, schema: Type[BaseDocument]) -> Dict[str, _Column]:
        columns: Dict[str, _Column] = dict()
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
                        for nested_name, t in self._create_columns(t_).items()
                    },
                )
            else:
                columns[field_name] = self._create_single_column(field)
        return columns

    def _create_single_column(self, field):
        type_ = field.type_
        db_type = self.python_type_to_db_type(type_)
        config = self._default_column_config[db_type].copy()
        custom_config = field.field_info.extra
        config.update(custom_config)
        # parse n_dim from parametrized tensor type
        if (
            hasattr(type_, '__docarray_target_shape__')
            and type_.__docarray_target_shape__
        ):
            if len(type_.__docarray_target_shape__) == 1:
                n_dim = type_.__docarray_target_shape__[0]
            else:
                n_dim = type_.__docarray_target_shape__
        else:
            n_dim = None
        return _Column(docarray_type=type_, db_type=db_type, config=config, n_dim=n_dim)

    def _is_schema_compatible(self, docs: Sequence[BaseDocument]) -> bool:
        """Flatten a DocumentArray into a DocumentArray of the schema type."""
        reference_col_db_types = [
            (name, col.db_type) for name, col in self._columns.items()
        ]
        if isinstance(docs, AnyDocumentArray):
            input_columns = self._create_columns(docs.document_type)
            input_col_db_types = [
                (name, col.db_type) for name, col in input_columns.items()
            ]
            # this could be relaxed in the future,
            # see schema translation ideas in the design doc
            return reference_col_db_types == input_col_db_types
        else:
            for d in docs:
                input_columns = self._create_columns(type(d))
                input_col_db_types = [
                    (name, col.db_type) for name, col in input_columns.items()
                ]
                # this could be relaxed in the future,
                # see schema translation ideas in the design doc
                if reference_col_db_types != input_col_db_types:
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
            for col_name in self._columns
        }
