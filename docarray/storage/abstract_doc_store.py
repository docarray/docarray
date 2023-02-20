from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
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

TSchema = TypeVar('TSchema', bound=BaseDocument)


class FindResultBatched(NamedTuple):
    documents: List[DocumentArray]
    scores: np.ndarray


@dataclass
class BaseDBConfig(ABC):
    ...


@dataclass
class BaseRuntimeConfig(ABC):
    # default configurations for every column type
    # a dictionary from a column type (DB specific) to a dictionary
    # of default configurations for that type
    # These configs are used if no configs are specified in the `Field(...)`
    # of a field in the Document schema (`cls._schema`)
    # Example: `default_column_config['VARCHAR'] = {'length': 255}`
    default_column_config: Dict[Type, Dict[str, Any]]


@dataclass
class _Column:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]


def _delegate_to_query(method_name: str):
    def inner(self, *args, **kwargs):
        if args:
            raise ValueError(
                f'Positional arguments are not supported for '
                f'{type(self)}.`{method_name}`.'
                f' Use keyword arguments instead.'
            )
        self._queries.append((method_name, kwargs))
        return self

    return inner


class BaseQueryBuilder(ABC):
    def __init__(self):
        # list of tuples (method name, kwargs)
        self._queries: List[Tuple[str, Dict]] = []

    find = _delegate_to_query('find')
    find_batched = _delegate_to_query('find_batched')
    filter = _delegate_to_query('filter')
    filter_batched = _delegate_to_query('filter_batched')
    text_search = _delegate_to_query('text_search')
    text_search_batched = _delegate_to_query('text_search_batched')

    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        """Build the DB specific query object.
        The DB specific implementation can leverage self._queries to do so.
        The output of this should be able to be passed to execute_query().
        """
        ...


class BaseDocumentStore(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    # the BaseDocument that defines the schema of the store
    # for subclasses this is filled automatically
    _schema: Optional[Type[BaseDocument]] = None

    # register helper classes here
    _query_builder_cls: Type[BaseQueryBuilder] = BaseQueryBuilder
    _db_config_cls: Type  # should be dataclass
    _runtime_config_cls: Type  # should be dataclass

    def __init__(self, db_config=None, **kwargs):
        if self._schema is None:
            raise ValueError(
                'A DocumentStore must be typed with a Document type.'
                'To do so, use the syntax: DocumentStore[DocumentType]'
            )
        self._db_config = db_config if db_config else self._db_config_cls(**kwargs)
        if not isinstance(self._db_config, self._db_config_cls):
            raise ValueError(f'db_config must be of type {self._db_config_cls}')
        self._runtime_config = self._runtime_config_cls()
        self._columns: Dict[str, _Column] = self._create_columns(self._schema)

    #####################################
    # Abstract methods                  #
    # Subclasses must implement these   #
    #####################################
    @abstractmethod
    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        ...

    @abstractmethod
    def index(self, docs: Union[TSchema, Sequence[TSchema]]):
        """Index a document into the store"""
        ...

    @abstractmethod
    def num_docs(self) -> int:
        """Return the number of indexed documents"""
        ...

    @abstractmethod
    def __delitem__(self, key: Union[str, Sequence[str]]):
        """Delete one or multiple Documents from the store, by `id`.

        :param key: id or ids to delete from the Document Store
        """
        ...

    @abstractmethod
    def __getitem__(self, key: Union[str, Sequence[str]]):
        """Get one or multiple Documents into the store, by `id`.

        :param key: id or ids to get from the Document Store
        """
        ...

    @abstractmethod
    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Execute a query on the database.
        This is intended as a pass-through to the underlying database, so that users
        can enjoy anything that is not available through our API.

        Also, this is the method that the output of the query builder is passed to.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
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

    @abstractmethod
    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocumentArray:
        """Find documents in the store based on a filter query

        :param filter_query: the DB specific filter query to execute
        """
        # TODO(johannes) refine method signature
        ...

    @abstractmethod
    def filter_batched(
        self,
        filter_queries: Any,
        limit: int = 10,
        **kwargs,
    ) -> List[DocumentArray]:
        """Find documents in the store based on multiple filter queries

        :param filter_queries: the DB specific filter queries to execute
        """
        # TODO(johannes) refine method signature
        ...

    @abstractmethod
    def text_search(
        self,
        query: str,
        embedding_field: str = 'embedding',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the store based on a text search query

        :param query: The text to search for
        """
        # TODO(johannes) refine method signature
        ...

    @abstractmethod
    def text_search_batched(
        self,
        queries: List[str],
        embedding_field: str = 'embedding',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find documents in the store based on a text search query

        :param query: The text to search for
        """
        # TODO(johannes) refine method signature
        ...

    ####################################################
    # Optional overrides                               #
    # Subclasses may more may not need to change these #
    ####################################################

    def configure(self, runtime_config=None, **kwargs):
        """
        Configure the DocumentStore.
        You can either pass a config object to `config` or pass individual config
        parameters as keyword arguments.
        If a configuration object is passed, it will replace the current configuration.
        If keyword arguments are passed, they will update the current configuration.

        :param runtime_config: the configuration to apply
        :param kwargs: individual configuration parameters
        """
        if runtime_config is None:
            self._runtime_config = replace(self._runtime_config, **kwargs)
        else:
            if not isinstance(runtime_config, self._runtime_config_cls):
                raise ValueError(
                    f'runtime_config must be of type {self._runtime_config_cls}'
                )
            self._runtime_config = runtime_config

    ##################################################
    # Behind-the-scenes magic                        #
    # Subclasses should not need to implement these  #
    ##################################################
    def __class_getitem__(cls, item: Type[TSchema]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] `item` should be a Document not a {item} '
            )

        class _DocumentStoreTyped(cls):  # type: ignore
            _schema: Type[TSchema] = item

        _DocumentStoreTyped.__name__ = f'{cls.__name__}[{item.__name__}]'
        _DocumentStoreTyped.__qualname__ = f'{cls.__qualname__}[{item.__name__}]'

        return _DocumentStoreTyped

    def build_query(self) -> BaseQueryBuilder:
        """
        Build a query for this DocumentStore.

        :return: a new `QueryBuilder` object for this DocumentStore
        """
        return self._query_builder_cls()

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
        config = self._runtime_config.default_column_config[db_type].copy()
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

        def _col_gen(col_name):
            return (self.get_value(doc, col_name) for doc in docs)

        return {col_name: _col_gen(col_name) for col_name in self._columns}
