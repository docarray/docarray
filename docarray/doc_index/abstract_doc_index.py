from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from typing_inspect import is_union_type

from docarray import BaseDocument, DocumentArray
from docarray.array.abstract_array import AnyDocumentArray
from docarray.typing import AnyTensor
from docarray.utils.find import FindResult, _FindResult
from docarray.utils.misc import torch_imported

if TYPE_CHECKING:
    from pydantic.fields import ModelField

if torch_imported:
    import torch

TSchema = TypeVar('TSchema', bound=BaseDocument)


class FindResultBatched(NamedTuple):
    documents: List[DocumentArray]
    scores: np.ndarray


class _FindResultBatched(NamedTuple):
    documents: Union[List[DocumentArray], List[List[Dict[str, Any]]]]
    scores: np.ndarray


def _raise_not_composable(name):
    def _inner(self, *args, **kwargs):
        raise NotImplementedError(
            f'`{name}` is not usable through the query builder of this Document Index ({type(self)}). '
            f'But you can call `{type(self)}.{name}()` directly.'
        )

    return _inner


def _raise_not_supported(name):
    def _inner(self, *args, **kwargs):
        raise NotImplementedError(
            f'`{name}` is not usable through the query builder of this Document Index ({type(self)}). '
        )

    return _inner


@dataclass
class _ColumnInfo:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]


class BaseDocumentIndex(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    # the BaseDocument that defines the schema of the store
    # for subclasses this is filled automatically
    _schema: Optional[Type[BaseDocument]] = None

    def __init__(self, db_config=None, **kwargs):
        if self._schema is None:
            raise ValueError(
                'A DocumentIndex must be typed with a Document type.'
                'To do so, use the syntax: DocumentIndex[DocumentType]'
            )
        self._db_config = db_config or self.DBConfig(**kwargs)
        if not isinstance(self._db_config, self.DBConfig):
            raise ValueError(f'db_config must be of type {self.DBConfig}')
        self._runtime_config = self.RuntimeConfig()
        self._column_infos: Dict[str, _ColumnInfo] = self._create_columns(self._schema)

    ###############################################
    # Inner classes for query builder and configs #
    # Subclasses must subclass & implement these  #
    ###############################################

    class QueryBuilder(ABC):
        @abstractmethod
        def build(self, *args, **kwargs) -> Any:
            """Build the DB specific query object.
            The DB specific implementation can leverage self._queries to do so.
            The output of this should be able to be passed to execute_query().
            """
            ...

        # the methods below need to be implemented by subclasses
        # If, in your subclass, one of these is not usable in a query builder, but
        # can be called directly on the DocumentIndex, use `_raise_not_composable`.
        # If the method is not supported _at all_, use `_raise_not_supported`.
        find = abstractmethod(lambda *args, **kwargs: ...)
        filter = abstractmethod(lambda *args, **kwargs: ...)
        text_search = abstractmethod(lambda *args, **kwargs: ...)
        find_batched = abstractmethod(lambda *args, **kwargs: ...)
        filter_batched = abstractmethod(lambda *args, **kwargs: ...)
        text_search_batched = abstractmethod(lambda *args, **kwargs: ...)

    @dataclass
    class DBConfig(ABC):
        ...

    @dataclass
    class RuntimeConfig(ABC):
        # default configurations for every column type
        # a dictionary from a column type (DB specific) to a dictionary
        # of default configurations for that type
        # These configs are used if no configs are specified in the `Field(...)`
        # of a field in the Document schema (`cls._schema`)
        # Example: `default_column_config['VARCHAR'] = {'length': 255}`
        default_column_config: Dict[Type, Dict[str, Any]] = field(default_factory=dict)

    #####################################
    # Abstract methods                  #
    # Subclasses must implement these   #
    #####################################

    @abstractmethod
    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        ...

    @abstractmethod
    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        """Index a document into the store"""
        # `column_to_data` is a dictionary from column name to a generator
        # that yields the data for that column.
        # If you want to work directly on documents, you can implement index() instead
        # If you implement index(), _index() only needs a dummy implementation.
        ...

    @abstractmethod
    def num_docs(self) -> int:
        """Return the number of indexed documents"""
        ...

    @abstractmethod
    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        ...

    @abstractmethod
    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document Index
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`. Duplicate `doc_ids` can be omitted in the output.
        """
        ...

    @abstractmethod
    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Execute a query on the database.

        Can take two kinds of inputs:
        - A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document Index API.
        - The output of this Document Index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        ...

    @abstractmethod
    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        """Find documents in the index

        :param query: query vector for KNN/ANN search. Has single axis.
        :param limit: maximum number of documents to return per query
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        ...

    @abstractmethod
    def _find_batched(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        """Find documents in the index

        :param query: query vectors for KNN/ANN search.
            Has shape (batch_size, vector_dim)
        :param limit: maximum number of documents to return
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        ...

    @abstractmethod
    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> Union[DocumentArray, List[Dict]]:
        """Find documents in the index based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocumentArray containing the documents that match the filter query
        """
        ...

    @abstractmethod
    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> Union[List[DocumentArray], List[List[Dict]]]:
        """Find documents in the index based on multiple filter queries.
        Each query is considered individually, and results are returned per query.

        :param filter_queries: the DB specific filter queries to execute
        :param limit: maximum number of documents to return per query
        :return: List of DocumentArrays containing the documents
            that match the filter queries
        """
        ...

    @abstractmethod
    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        """Find documents in the index based on a text search query

        :param query: The text to search for
        :param limit: maximum number of documents to return
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        ...

    @abstractmethod
    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        """Find documents in the index based on a text search query

        :param queries: The texts to search for
        :param limit: maximum number of documents to return per query
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        ...

    ####################################################
    # Optional overrides                               #
    # Subclasses may or may not need to change these #
    ####################################################

    def __getitem__(
        self, key: Union[str, Sequence[str]]
    ) -> Union[TSchema, DocumentArray[TSchema]]:
        """Get one or multiple Documents into the index, by `id`.
        If no document is found, a KeyError is raised.

        :param key: id or ids to get from the Document Index
        """
        # normalize input
        if isinstance(key, str):
            return_singleton = True
            key = [key]
        else:
            return_singleton = False
        # retrieve data
        doc_sequence = self._get_items(key)
        # check data
        if len(doc_sequence) == 0:
            raise KeyError(f'No document with id {key} found')

        # cast output
        if isinstance(doc_sequence, DocumentArray):
            out_da: DocumentArray[TSchema] = doc_sequence
        elif isinstance(doc_sequence[0], Dict):
            out_da = self._dict_list_to_docarray(doc_sequence)  # type: ignore
        else:
            da_cls = DocumentArray.__class_getitem__(
                cast(Type[BaseDocument], self._schema)
            )
            out_da = da_cls(doc_sequence)

        return out_da[0] if return_singleton else out_da

    def __delitem__(self, key: Union[str, Sequence[str]]):
        """Delete one or multiple Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param key: id or ids to delete from the Document Index
        """
        if isinstance(key, str):
            key = [key]
        self._del_items(key)

    def configure(self, runtime_config=None, **kwargs):
        """
        Configure the DocumentIndex.
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
            if not isinstance(runtime_config, self.RuntimeConfig):
                raise ValueError(f'runtime_config must be of type {self.RuntimeConfig}')
            self._runtime_config = runtime_config

    def index(self, docs: Union[BaseDocument, Sequence[BaseDocument]], **kwargs):
        """Index Documents into the index.

        :param docs: Documents to index
        """
        data_by_columns = self._get_col_value_dict(docs)
        self._index(data_by_columns, **kwargs)

    def find(
        self,
        query: Union[AnyTensor, BaseDocument],
        search_field: str = 'embedding',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the index using nearest neighbor search.

        :param query: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.)
            with a single axis, or a Document
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        if isinstance(query, BaseDocument):
            query_vec = self._get_values_by_column([query], search_field)[0]
        else:
            query_vec = query
        query_vec_np = self._to_numpy(query_vec)
        docs, scores = self._find(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def find_batched(
        self,
        queries: Union[AnyTensor, DocumentArray],
        search_field: str = 'embedding',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find documents in the index using nearest neighbor search.

        :param queries: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.) with a,
            or a DocumentArray.
            If a tensor-like is passed, it should have shape (batch_size, vector_dim)
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return per query
        :return: a named tuple containing `documents` and `scores`
        """
        if isinstance(queries, Sequence):
            query_vec_list = self._get_values_by_column(queries, search_field)
            query_vec_np = np.stack(
                tuple(self._to_numpy(query_vec) for query_vec in query_vec_list)
            )
        else:
            query_vec_np = self._to_numpy(queries)

        da_list, scores = self._find_batched(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if len(da_list) > 0 and isinstance(da_list[0], List):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return FindResultBatched(documents=da_list, scores=scores)  # type: ignore

    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocumentArray:
        """Find documents in the index based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocumentArray containing the documents that match the filter query
        """
        docs = self._filter(filter_query, limit=limit, **kwargs)

        if isinstance(docs, List):
            docs = self._dict_list_to_docarray(docs)

        return docs

    def filter_batched(
        self,
        filter_queries: Any,
        limit: int = 10,
        **kwargs,
    ) -> List[DocumentArray]:
        """Find documents in the index based on multiple filter queries.

        :param filter_queries: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocumentArray containing the documents that match the filter query
        """
        da_list = self._filter_batched(filter_queries, limit=limit, **kwargs)

        if len(da_list) > 0 and isinstance(da_list[0], List):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return da_list  # type: ignore

    def text_search(
        self,
        query: Union[str, BaseDocument],
        search_field: str = 'text',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the index based on a text search query.

        :param query: The text to search for
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        if isinstance(query, BaseDocument):
            query_text = self._get_values_by_column([query], search_field)[0]
        else:
            query_text = query
        docs, scores = self._text_search(
            query_text, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def text_search_batched(
        self,
        queries: Union[Sequence[str], Sequence[BaseDocument]],
        search_field: str = 'text',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find documents in the index based on a text search query.

        :param queries: The texts to search for
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        if isinstance(queries[0], BaseDocument):
            query_docs: Sequence[BaseDocument] = cast(Sequence[BaseDocument], queries)
            query_texts: Sequence[str] = self._get_values_by_column(
                query_docs, search_field
            )
        else:
            query_texts = cast(Sequence[str], queries)
        da_list, scores = self._text_search_batched(
            query_texts, search_field=search_field, limit=limit, **kwargs
        )

        if len(da_list) > 0 and isinstance(da_list[0], List):
            docs = [self._dict_list_to_docarray(docs) for docs in da_list]
            return FindResultBatched(documents=docs, scores=scores)

        return FindResultBatched(documents=da_list, scores=scores)

    ##########################################################
    # Helper methods                                         #
    # These might be useful in your subclass implementation  #
    ##########################################################

    @staticmethod
    def _get_values_by_column(docs: Sequence[BaseDocument], col_name: str) -> List[Any]:
        """Get the value of a column of a document.

        :param docs: The DocumentArray to get the values from
        :param col_name: The name of the column, e.g. 'text' or 'image__tensor'
        :return: The value of the column of `doc`
        """
        leaf_vals = []
        for doc in docs:
            if '__' in col_name:
                fields = col_name.split('__')
                leaf_doc: BaseDocument = doc
                for f in fields[:-1]:
                    leaf_doc = getattr(leaf_doc, f)
                leaf_vals.append(getattr(leaf_doc, fields[-1]))
            else:
                leaf_vals.append(getattr(doc, col_name))
        return leaf_vals

    @staticmethod
    def _transpose_col_value_dict(
        col_value_dict: Dict[str, Iterable[Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """'Transpose' the output of `_get_col_value_dict()`: Yield rows of columns, where each row represent one Document.
        Since a generator is returned, this process comes at negligible cost.

        :param docs: The DocumentArray to get the values from
        :return: The `docs` flattened out as rows. Each row is a dictionary mapping from column name to value
        """
        return (dict(zip(col_value_dict, row)) for row in zip(*col_value_dict.values()))

    def _get_col_value_dict(
        self, docs: Union[BaseDocument, Sequence[BaseDocument]]
    ) -> Dict[str, Generator[Any, None, None]]:
        """
        Get all data from a (sequence of) document(s), flattened out by column.
        This can be seen as the transposed representation of `_get_rows()`.

        :param docs: The document(s) to get the data from
        :return: A dictionary mapping column names to a generator of values
        """
        if isinstance(docs, BaseDocument):
            docs_seq: Sequence[BaseDocument] = [docs]
        else:
            docs_seq = docs
        if not self._is_schema_compatible(docs_seq):
            raise ValueError(
                'The schema of the documents to be indexed is not compatible'
                ' with the schema of the index.'
            )

        def _col_gen(col_name: str):
            return (self._get_values_by_column([doc], col_name)[0] for doc in docs_seq)

        return {col_name: _col_gen(col_name) for col_name in self._column_infos}

    ##################################################
    # Behind-the-scenes magic                        #
    # Subclasses should not need to implement these  #
    ##################################################
    def __class_getitem__(cls, item: Type[TSchema]):
        if not isinstance(item, type):
            # do nothing
            # enables use in static contexts with type vars, e.g. as type annotation
            return Generic.__class_getitem__.__func__(cls, item)
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] `item` should be a Document not a {item} '
            )

        class _DocumentIndexTyped(cls):  # type: ignore
            _schema: Type[TSchema] = item

        _DocumentIndexTyped.__name__ = f'{cls.__name__}[{item.__name__}]'
        _DocumentIndexTyped.__qualname__ = f'{cls.__qualname__}[{item.__name__}]'

        return _DocumentIndexTyped

    def build_query(self) -> QueryBuilder:
        """
        Build a query for this DocumentIndex.

        :return: a new `QueryBuilder` object for this DocumentIndex
        """
        return self.QueryBuilder()  # type: ignore

    def _create_columns(self, schema: Type[BaseDocument]) -> Dict[str, _ColumnInfo]:
        columns: Dict[str, _ColumnInfo] = dict()
        for field_name, field_ in schema.__fields__.items():
            t_ = schema._get_field_type(field_name)
            if is_union_type(t_):
                raise ValueError(
                    'Union types are not supported in the schema of a DocumentIndex.'
                    f' Instead of using type {t_} use a single specific type.'
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
                columns[field_name] = self._create_single_column(field_, t_)
        return columns

    def _create_single_column(self, field: 'ModelField', type_: Type) -> _ColumnInfo:
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
        return _ColumnInfo(
            docarray_type=type_, db_type=db_type, config=config, n_dim=n_dim
        )

    def _is_schema_compatible(self, docs: Sequence[BaseDocument]) -> bool:
        """Flatten a DocumentArray into a DocumentArray of the schema type."""
        reference_col_db_types = [
            (name, col.db_type) for name, col in self._column_infos.items()
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

    def _to_numpy(self, val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val
        elif isinstance(val, (list, tuple)):
            return np.array(val)
        elif torch_imported and isinstance(val, torch.Tensor):
            return val.numpy()
        else:
            raise ValueError(f'Unsupported input type for {type(self)}: {type(val)}')

    def _convert_dict_to_doc(
        self, doc_dict: Dict[str, Any], schema: Type[BaseDocument]
    ) -> BaseDocument:
        """
        Convert a dict to a Document object.

        :param doc_dict: A dict that contains all the flattened fields of a Document, the field names are the keys and follow the pattern {field_name} or {field_name}__{nested_name}
        :param schema: The schema of the Document object
        :return: A Document object
        """

        for field_name, _ in schema.__fields__.items():
            t_ = schema._get_field_type(field_name)
            if issubclass(t_, BaseDocument):
                inner_dict = {}

                fields = [
                    key for key in doc_dict.keys() if key.startswith(f'{field_name}__')
                ]
                for key in fields:
                    nested_name = key[len(f'{field_name}__') :]
                    inner_dict[nested_name] = doc_dict.pop(key)

                doc_dict[field_name] = self._convert_dict_to_doc(inner_dict, t_)

        schema_cls = cast(Type[BaseDocument], schema)
        return schema_cls(**doc_dict)

    def _dict_list_to_docarray(
        self, dict_list: Sequence[Dict[str, Any]]
    ) -> DocumentArray:
        """Convert a list of docs in dict type to a DocumentArray of the schema type."""

        doc_list = [self._convert_dict_to_doc(doc_dict, self._schema) for doc_dict in dict_list]  # type: ignore
        da_cls = DocumentArray.__class_getitem__(cast(Type[BaseDocument], self._schema))
        return da_cls(doc_list)
