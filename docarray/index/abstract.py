import copy
import logging
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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from pydantic.error_wrappers import ValidationError
from typing_inspect import get_args, is_optional_type, is_union_type

from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.typing import ID, AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import is_tensor_union, safe_issubclass
from docarray.utils._internal.misc import import_library
from docarray.utils.find import (
    FindResult,
    FindResultBatched,
    SubindexFindResult,
    _FindResult,
    _FindResultBatched,
)

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
    from pydantic.fields import ModelField

    from docarray.typing import TensorFlowTensor
else:
    tf = import_library('tensorflow', raise_error=False)
    if tf is not None:
        from docarray.typing import TensorFlowTensor
    torch = import_library('torch', raise_error=False)

TSchema = TypeVar('TSchema', bound=BaseDoc)


def _raise_not_composable(name):
    def _inner(self, *args, **kwargs):
        raise NotImplementedError(
            f'`{name}` is not usable through the query builder of this Document index ({type(self)}). '
            f'But you can call `{type(self)}.{name}()` directly.'
        )

    return _inner


def _raise_not_supported(name):
    def _inner(self, *args, **kwargs):
        raise NotImplementedError(
            f'`{name}` is not usable through the query builder of this Document index ({type(self)}). '
        )

    return _inner


@dataclass
class _ColumnInfo:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]


class BaseDocIndex(ABC, Generic[TSchema]):
    """Abstract class for all Document Stores"""

    # the BaseDoc that defines the schema of the store
    # for subclasses this is filled automatically
    _schema: Optional[Type[BaseDoc]] = None

    def __init__(self, db_config=None, subindex: bool = False, **kwargs):
        if self._schema is None:
            raise ValueError(
                'A DocumentIndex must be typed with a Document type.'
                'To do so, use the syntax: DocumentIndex[DocumentType]'
            )
        if subindex:

            class _NewSchema(self._schema):  # type: ignore
                parent_id: Optional[ID] = None

            self._ori_schema = self._schema
            self._schema = cast(Type[BaseDoc], _NewSchema)

        self._logger = logging.getLogger('docarray')
        self._db_config = db_config or self.DBConfig(**kwargs)
        if not isinstance(self._db_config, self.DBConfig):
            raise ValueError(f'db_config must be of type {self.DBConfig}')
        self._logger.info('DB config created')
        self._runtime_config = self.RuntimeConfig()
        self._logger.info('Runtime config created')
        self._column_infos: Dict[str, _ColumnInfo] = self._create_column_infos(
            self._schema
        )
        self._is_subindex = subindex
        self._subindices: Dict[str, BaseDocIndex] = {}
        self._init_subindex()

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

        # TODO support subindex in QueryBuilder

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
        index_name: Optional[str] = None
        # default configurations for every column type
        # a dictionary from a column type (DB specific) to a dictionary
        # of default configurations for that type
        # These configs are used if no configs are specified in the `Field(...)`
        # of a field in the Document schema (`cls._schema`)
        # Example: `default_column_config['VARCHAR'] = {'length': 255}`
        default_column_config: Dict[Type, Dict[str, Any]] = field(default_factory=dict)

    @dataclass
    class RuntimeConfig(ABC):
        pass

    @property
    def index_name(self):
        """Return the name of the index in the database."""
        ...

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
        """index a document into the store"""
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

        :param doc_ids: ids to get from the Document index
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`. Duplicate `doc_ids` can be omitted in the output.
        """
        ...

    @abstractmethod
    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Execute a query on the database.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        ...

    @abstractmethod
    def _doc_exists(self, doc_id: str) -> bool:
        """
        Checks if a given document exists in the index.

        :param doc_id: The id of a document to check.
        :return: True if the document exists in the index, False otherwise.
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
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        """Find documents in the index

        :param queries: query vectors for KNN/ANN search.
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
    ) -> Union[DocList, List[Dict]]:
        """Find documents in the index based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        ...

    @abstractmethod
    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> Union[List[DocList], List[List[Dict]]]:
        """Find documents in the index based on multiple filter queries.
        Each query is considered individually, and results are returned per query.

        :param filter_queries: the DB specific filter queries to execute
        :param limit: maximum number of documents to return per query
        :return: List of DocLists containing the documents that match the filter
            queries
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
    ) -> Union[TSchema, DocList[TSchema]]:
        """Get one or multiple Documents into the index, by `id`.
        If no document is found, a KeyError is raised.

        :param key: id or ids to get from the Document index
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

        # retrieve nested data
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self._schema)
        ):
            if safe_issubclass(type_, AnyDocArray) and isinstance(
                doc_sequence[0], Dict
            ):
                for doc in doc_sequence:
                    self._get_subindex_doclist(doc, field_name)  # type: ignore

        # cast output
        if isinstance(doc_sequence, DocList):
            out_docs: DocList[TSchema] = doc_sequence
        elif isinstance(doc_sequence[0], Dict):
            out_docs = self._dict_list_to_docarray(doc_sequence)  # type: ignore
        else:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
            out_docs = docs_cls(doc_sequence)

        return out_docs[0] if return_singleton else out_docs

    def __delitem__(self, key: Union[str, Sequence[str]]):
        """Delete one or multiple Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param key: id or ids to delete from the Document index
        """
        self._logger.info(f'Deleting documents with id(s) {key} from the index')
        if isinstance(key, str):
            key = [key]

        # delete nested data
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self._schema)
        ):
            if safe_issubclass(type_, AnyDocArray):
                for doc_id in key:
                    nested_docs_id = self._subindices[field_name]._filter_by_parent_id(
                        doc_id
                    )
                    if nested_docs_id:
                        del self._subindices[field_name][nested_docs_id]
        # delete data
        self._del_items(key)

    def __contains__(self, item: BaseDoc) -> bool:
        """
        Checks if a given document exists in the index.

        :param item: The document to check.
            It must be an instance of BaseDoc or its subclass.
        :return: True if the document exists in the index, False otherwise.
        """
        if safe_issubclass(type(item), BaseDoc):
            return self._doc_exists(str(item.id))
        else:
            raise TypeError(
                f"item must be an instance of BaseDoc or its subclass, not '{type(item).__name__}'"
            )

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

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
        """index Documents into the index.

        !!! note
            Passing a sequence of Documents that is not a DocList
            (such as a List of Docs) comes at a performance penalty.
            This is because the Index needs to check compatibility between itself and
            the data. With a DocList as input this is a single check; for other inputs
            compatibility needs to be checked for every Document individually.

        :param docs: Documents to index.
        """
        n_docs = 1 if isinstance(docs, BaseDoc) else len(docs)
        self._logger.debug(f'Indexing {n_docs} documents')
        docs_validated = self._validate_docs(docs)
        self._update_subindex_data(docs_validated)
        data_by_columns = self._get_col_value_dict(docs_validated)
        self._index(data_by_columns, **kwargs)

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
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
        self._logger.debug(f'Executing `find` for search field {search_field}')

        self._validate_search_field(search_field)
        if isinstance(query, BaseDoc):
            query_vec = self._get_values_by_column([query], search_field)[0]
        else:
            query_vec = query
        query_vec_np = self._to_numpy(query_vec)
        docs, scores = self._find(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def find_subindex(
        self,
        query: Union[AnyTensor, BaseDoc],
        subindex: str = '',
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> SubindexFindResult:
        """Find documents in subindex level.

        :param query: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.)
            with a single axis, or a Document
        :param subindex: name of the subindex to search on
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return
        :return: a named tuple containing root docs, subindex docs and scores
        """
        self._logger.debug(f'Executing `find_subindex` for search field {search_field}')

        sub_docs, scores = self._find_subdocs(
            query, subindex=subindex, search_field=search_field, limit=limit, **kwargs
        )

        fields = subindex.split('__')
        root_ids = [
            self._get_root_doc_id(doc.id, fields[0], '__'.join(fields[1:]))
            for doc in sub_docs
        ]
        root_docs = DocList[self._schema]()  # type: ignore
        for id in root_ids:
            root_docs.append(self[id])

        return SubindexFindResult(
            root_documents=root_docs, sub_documents=sub_docs, scores=scores  # type: ignore
        )

    def find_batched(
        self,
        queries: Union[AnyTensor, DocList],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find documents in the index using nearest neighbor search.

        :param queries: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.) with a,
            or a DocList.
            If a tensor-like is passed, it should have shape (batch_size, vector_dim)
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return per query
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `find_batched` for search field {search_field}')

        if search_field:
            if '__' in search_field:
                fields = search_field.split('__')
                if safe_issubclass(self._schema._get_field_type(fields[0]), AnyDocArray):  # type: ignore
                    return self._subindices[fields[0]].find_batched(
                        queries,
                        search_field='__'.join(fields[1:]),
                        limit=limit,
                        **kwargs,
                    )

        self._validate_search_field(search_field)
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
        if (
            len(da_list) > 0
            and isinstance(da_list[0], List)
            and not isinstance(da_list[0], DocList)
        ):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return FindResultBatched(documents=da_list, scores=scores)  # type: ignore

    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocList:
        """Find documents in the index based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        self._logger.debug(f'Executing `filter` for the query {filter_query}')
        docs = self._filter(filter_query, limit=limit, **kwargs)

        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return docs

    def filter_subindex(
        self,
        filter_query: Any,
        subindex: str,
        limit: int = 10,
        **kwargs,
    ) -> DocList:
        """Find documents in subindex level based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param subindex: name of the subindex to search on
        :param limit: maximum number of documents to return
        :return: a DocList containing the subindex level documents that match the filter query
        """
        self._logger.debug(
            f'Executing `filter` for the query {filter_query} in subindex {subindex}'
        )
        if '__' in subindex:
            fields = subindex.split('__')
            return self._subindices[fields[0]].filter_subindex(
                filter_query, '__'.join(fields[1:]), limit=limit, **kwargs
            )
        else:
            return self._subindices[subindex].filter(
                filter_query, limit=limit, **kwargs
            )

    def filter_batched(
        self,
        filter_queries: Any,
        limit: int = 10,
        **kwargs,
    ) -> List[DocList]:
        """Find documents in the index based on multiple filter queries.

        :param filter_queries: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        self._logger.debug(
            f'Executing `filter_batched` for the queries {filter_queries}'
        )
        da_list = self._filter_batched(filter_queries, limit=limit, **kwargs)

        if len(da_list) > 0 and isinstance(da_list[0], List):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return da_list  # type: ignore

    def text_search(
        self,
        query: Union[str, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the index based on a text search query.

        :param query: The text to search for
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `text_search` for search field {search_field}')
        self._validate_search_field(search_field)
        if isinstance(query, BaseDoc):
            query_text = self._get_values_by_column([query], search_field)[0]
        else:
            query_text = query
        docs, scores = self._text_search(
            query_text, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def text_search_batched(
        self,
        queries: Union[Sequence[str], Sequence[BaseDoc]],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find documents in the index based on a text search query.

        :param queries: The texts to search for
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(
            f'Executing `text_search_batched` for search field {search_field}'
        )
        self._validate_search_field(search_field)
        if isinstance(queries[0], BaseDoc):
            query_docs: Sequence[BaseDoc] = cast(Sequence[BaseDoc], queries)
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

        da_list_ = cast(List[DocList], da_list)
        return FindResultBatched(documents=da_list_, scores=scores)

    def _filter_by_parent_id(self, id: str) -> Optional[List[str]]:
        """Filter the ids of the subindex documents given id of root document.

        :param id: the root document id to filter by
        :return: a list of ids of the subindex documents
        """
        return None

    ##########################################################
    # Helper methods                                         #
    # These might be useful in your subclass implementation  #
    ##########################################################

    @staticmethod
    def _get_values_by_column(docs: Sequence[BaseDoc], col_name: str) -> List[Any]:
        """Get the value of a column of a document.

        :param docs: The DocList to get the values from
        :param col_name: The name of the column, e.g. 'text' or 'image__tensor'
        :return: The value of the column of `doc`
        """
        leaf_vals = []
        for doc in docs:
            if '__' in col_name:
                fields = col_name.split('__')
                leaf_doc: BaseDoc = doc
                for f in fields[:-1]:
                    leaf_doc = getattr(leaf_doc, f)
                leaf_vals.append(getattr(leaf_doc, fields[-1]))
            else:
                leaf_vals.append(getattr(doc, col_name))
        return leaf_vals

    @staticmethod
    def _transpose_col_value_dict(
        col_value_dict: Mapping[str, Iterable[Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """'Transpose' the output of `_get_col_value_dict()`: Yield rows of columns, where each row represent one Document.
        Since a generator is returned, this process comes at negligible cost.

        :param docs: The DocList to get the values from
        :return: The `docs` flattened out as rows. Each row is a dictionary mapping from column name to value
        """
        return (dict(zip(col_value_dict, row)) for row in zip(*col_value_dict.values()))

    def _get_col_value_dict(
        self, docs: Union[BaseDoc, Sequence[BaseDoc]]
    ) -> Dict[str, Generator[Any, None, None]]:
        """
        Get all data from a (sequence of) document(s), flattened out by column.
        This can be seen as the transposed representation of `_get_rows()`.

        :param docs: The document(s) to get the data from
        :return: A dictionary mapping column names to a generator of values
        """
        if isinstance(docs, BaseDoc):
            docs_seq: Sequence[BaseDoc] = [docs]
        else:
            docs_seq = docs

        def _col_gen(col_name: str):
            return (
                self._to_numpy(
                    self._get_values_by_column([doc], col_name)[0],
                    allow_passthrough=True,
                )
                for doc in docs_seq
            )

        return {col_name: _col_gen(col_name) for col_name in self._column_infos}

    def _update_subindex_data(
        self,
        docs: DocList[BaseDoc],
    ):
        """
        Add `parent_id` to all sublevel documents.

        :param docs: The document(s) to update the `parent_id` for
        """
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self._schema)
        ):
            if safe_issubclass(type_, AnyDocArray):
                for doc in docs:
                    _list = getattr(doc, field_name)
                    for i, nested_doc in enumerate(_list):
                        nested_doc = self._subindices[field_name]._schema(  # type: ignore
                            **nested_doc.__dict__
                        )
                        nested_doc.parent_id = doc.id
                        _list[i] = nested_doc

    ##################################################
    # Behind-the-scenes magic                        #
    # Subclasses should not need to implement these  #
    ##################################################
    def __class_getitem__(cls, item: Type[TSchema]):
        if not isinstance(item, type):
            # do nothing
            # enables use in static contexts with type vars, e.g. as type annotation
            return Generic.__class_getitem__.__func__(cls, item)
        if not safe_issubclass(item, BaseDoc):
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

    @classmethod
    def _flatten_schema(
        cls, schema: Type[BaseDoc], name_prefix: str = ''
    ) -> List[Tuple[str, Type, 'ModelField']]:
        """Flatten the schema of a Document into a list of column names and types.
        Nested Documents are handled in a recursive manner by adding `'__'` as a prefix to the column name.

        :param schema: The schema to flatten
        :param name_prefix: prefix to append to the column names. Used for recursive calls to handle nesting.
        :return: A list of column names, types, and fields
        """
        names_types_fields: List[Tuple[str, Type, 'ModelField']] = []
        for field_name, field_ in schema.__fields__.items():
            t_ = schema._get_field_type(field_name)
            inner_prefix = name_prefix + field_name + '__'

            if is_union_type(t_):
                union_args = get_args(t_)

                if is_tensor_union(t_):
                    names_types_fields.append(
                        (name_prefix + field_name, AbstractTensor, field_)
                    )

                elif len(union_args) == 2 and type(None) in union_args:
                    # simple "Optional" type, treat as special case:
                    # treat as if it was a single non-optional type
                    for t_arg in union_args:
                        if t_arg is not type(None):
                            if safe_issubclass(t_arg, BaseDoc):
                                names_types_fields.extend(
                                    cls._flatten_schema(t_arg, name_prefix=inner_prefix)
                                )
                            else:
                                names_types_fields.append(
                                    (name_prefix + field_name, t_arg, field_)
                                )
                else:
                    raise ValueError(
                        f'Union type {t_} is not supported. Only Union of subclasses of AbstractTensor or Union[type, None] are supported.'
                    )
            elif safe_issubclass(t_, BaseDoc):
                names_types_fields.extend(
                    cls._flatten_schema(t_, name_prefix=inner_prefix)
                )
            elif safe_issubclass(t_, AbstractTensor):
                names_types_fields.append(
                    (name_prefix + field_name, AbstractTensor, field_)
                )
            else:
                names_types_fields.append((name_prefix + field_name, t_, field_))
        return names_types_fields

    def _create_column_infos(self, schema: Type[BaseDoc]) -> Dict[str, _ColumnInfo]:
        """Collects information about every column that is implied by a given schema.

        :param schema: The schema (subclass of BaseDoc) to analyze and parse
            columns from
        :returns: A dictionary mapping from column names to column information.
        """
        column_infos: Dict[str, _ColumnInfo] = dict()
        for field_name, type_, field_ in self._flatten_schema(schema):
            # Union types are handle in _flatten_schema
            if safe_issubclass(type_, AnyDocArray):
                column_infos[field_name] = _ColumnInfo(
                    docarray_type=type_, db_type=None, config=dict(), n_dim=None
                )
            else:
                column_infos[field_name] = self._create_single_column(field_, type_)

        return column_infos

    def _create_single_column(self, field: 'ModelField', type_: Type) -> _ColumnInfo:
        custom_config = field.field_info.extra
        if 'col_type' in custom_config.keys():
            db_type = custom_config['col_type']
            custom_config.pop('col_type')
            if db_type not in self._db_config.default_column_config.keys():
                raise ValueError(
                    f'The given col_type is not a valid db type: {db_type}'
                )
        else:
            db_type = self.python_type_to_db_type(type_)

        config = self._db_config.default_column_config[db_type].copy()
        config.update(custom_config)
        # parse n_dim from parametrized tensor type
        if (
            hasattr(field.type_, '__docarray_target_shape__')
            and field.type_.__docarray_target_shape__
        ):
            if len(field.type_.__docarray_target_shape__) == 1:
                n_dim = field.type_.__docarray_target_shape__[0]
            else:
                n_dim = field.type_.__docarray_target_shape__
        else:
            n_dim = None
        return _ColumnInfo(
            docarray_type=type_, db_type=db_type, config=config, n_dim=n_dim
        )

    def _init_subindex(
        self,
    ):
        """Initialize subindices if any column is subclass of AnyDocArray."""
        for col_name, col in self._column_infos.items():
            if safe_issubclass(col.docarray_type, AnyDocArray):
                sub_db_config = copy.deepcopy(self._db_config)
                sub_db_config.index_name = f'{self.index_name}__{col_name}'
                self._subindices[col_name] = self.__class__[col.docarray_type.doc_type](  # type: ignore
                    db_config=sub_db_config, subindex=True
                )

    def _validate_docs(
        self, docs: Union[BaseDoc, Sequence[BaseDoc]]
    ) -> DocList[BaseDoc]:
        """Validates Document against the schema of the Document Index.
        For validation to pass, the schema of `docs` and the schema of the Document
        Index need to evaluate to the same flattened columns.
        If Validation fails, a ValueError is raised.

        :param docs: Document to evaluate. If this is a DocList, validation is
            performed using its `doc_type` (parametrization), without having to check
            ever Document in `docs`. If this check fails, or if `docs` is not a
            DocList, evaluation is performed for every Document in `docs`.
        :return: A DocList containing the Documents in `docs`
        """
        if isinstance(docs, BaseDoc):
            docs = [docs]
        if isinstance(docs, DocList):
            # validation shortcut for DocList; only look at the schema
            reference_schema_flat = self._flatten_schema(
                cast(Type[BaseDoc], self._schema)
            )
            reference_names = [name for (name, _, _) in reference_schema_flat]
            reference_types = [t_ for (_, t_, _) in reference_schema_flat]
            try:
                input_schema_flat = self._flatten_schema(docs.doc_type)
            except ValueError:
                pass
            else:
                input_names = [name for (name, _, _) in input_schema_flat]
                input_types = [t_ for (_, t_, _) in input_schema_flat]
                # this could be relaxed in the future,
                # see schema translation ideas in the design doc
                names_compatible = reference_names == input_names
                types_compatible = all(
                    (safe_issubclass(t2, t1))
                    for (t1, t2) in zip(reference_types, input_types)
                )
                if names_compatible and types_compatible:
                    return docs

        out_docs = []
        for i in range(len(docs)):
            # validate the data
            try:
                out_docs.append(cast(Type[BaseDoc], self._schema).parse_obj(docs[i]))
            except (ValueError, ValidationError):
                raise ValueError(
                    'The schema of the input Documents is not compatible with the schema of the Document Index.'
                    ' Ensure that the field names of your data match the field names of the Document Index schema,'
                    ' and that the types of your data match the types of the Document Index schema.'
                )

        return DocList[BaseDoc].construct(out_docs)

    def _validate_search_field(self, search_field: Union[str, None]) -> bool:
        """
        Validate if the given `search_field` corresponds to one of the
        columns that was parsed from the schema.

        Some backends, like weaviate, don't use search fields, so the function
        returns True if `search_field` is empty or None.

        :param search_field: search field to validate.
        :return: True if the field exists, False otherwise.
        """
        if not search_field or search_field in self._column_infos.keys():
            if not search_field:
                self._logger.info('Empty search field was passed')
            return True
        else:
            valid_search_fields = ', '.join(self._column_infos.keys())
            raise ValueError(
                f'{search_field} is not a valid search field. Valid search fields are: {valid_search_fields}'
            )

    def _to_numpy(self, val: Any, allow_passthrough=False) -> Any:
        """
        Converts a value to a numpy array, if possible.

        :param val: The value to convert
        :param allow_passthrough: If True, the value is returned as-is if it is not convertible to a numpy array.
            If False, a `ValueError` is raised if the value is not convertible to a numpy array.
        :return: The value as a numpy array, or as-is if `allow_passthrough` is True and the value is not convertible
        """
        if isinstance(val, np.ndarray):
            return val
        if tf is not None and isinstance(val, TensorFlowTensor):
            return val.unwrap().numpy()
        if isinstance(val, (list, tuple)):
            return np.array(val)
        if torch is not None and isinstance(val, torch.Tensor):
            return val.detach().numpy()
        if tf is not None and isinstance(val, tf.Tensor):
            return val.numpy()
        if allow_passthrough:
            return val
        raise ValueError(f'Unsupported input type for {type(self)}: {type(val)}')

    def _convert_dict_to_doc(
        self, doc_dict: Dict[str, Any], schema: Type[BaseDoc], inner=False
    ) -> BaseDoc:
        """
        Convert a dict to a Document object.

        :param doc_dict: A dict that contains all the flattened fields of a Document, the field names are the keys and follow the pattern {field_name} or {field_name}__{nested_name}
        :param schema: The schema of the Document object
        :return: A Document object
        """
        for field_name, _ in schema.__fields__.items():
            t_ = schema._get_field_type(field_name)

            if not is_union_type(t_) and safe_issubclass(t_, AnyDocArray):
                self._get_subindex_doclist(doc_dict, field_name)

            if is_optional_type(t_):
                for t_arg in get_args(t_):
                    if t_arg is not type(None):
                        t_ = t_arg

            if not is_union_type(t_) and safe_issubclass(t_, BaseDoc):
                inner_dict = {}

                fields = [
                    key for key in doc_dict.keys() if key.startswith(f'{field_name}__')
                ]
                for key in fields:
                    nested_name = key[len(f'{field_name}__') :]
                    inner_dict[nested_name] = doc_dict.pop(key)

                doc_dict[field_name] = self._convert_dict_to_doc(
                    inner_dict, t_, inner=True
                )

        if self._is_subindex and not inner:
            doc_dict.pop('parent_id', None)
            schema_cls = cast(Type[BaseDoc], self._ori_schema)
        else:
            schema_cls = cast(Type[BaseDoc], schema)
        doc = schema_cls(**doc_dict)
        return doc

    def _dict_list_to_docarray(self, dict_list: Sequence[Dict[str, Any]]) -> DocList:
        """Convert a list of docs in dict type to a DocList of the schema type."""
        doc_list = [self._convert_dict_to_doc(doc_dict, self._schema) for doc_dict in dict_list]  # type: ignore
        if self._is_subindex:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._ori_schema))
        else:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
        return docs_cls(doc_list)

    def __len__(self) -> int:
        return self.num_docs()

    def _index_subindex(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        """Index subindex documents in the corresponding subindex.

        :param column_to_data: A dictionary from column name to a generator
        """
        for col_name, col in self._column_infos.items():
            if safe_issubclass(col.docarray_type, AnyDocArray):
                docs = [
                    doc for doc_list in column_to_data[col_name] for doc in doc_list
                ]
                self._subindices[col_name].index(docs)
                column_to_data.pop(col_name, None)

    def _get_subindex_doclist(self, doc: Dict[str, Any], field_name: str):
        """Get subindex Documents from the index and assign them to `field_name`.

        :param doc: a dictionary mapping from column name to value
        :param field_name: field name of the subindex Documents
        """
        if field_name not in doc.keys():
            parent_id = doc['id']
            nested_docs_id = self._subindices[field_name]._filter_by_parent_id(
                parent_id
            )
            if nested_docs_id:
                doc[field_name] = self._subindices[field_name].__getitem__(
                    nested_docs_id
                )

    def _find_subdocs(
        self,
        query: Union[AnyTensor, BaseDoc],
        subindex: str = '',
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the subindex and return subindex docs and scores."""
        fields = subindex.split('__')
        if not subindex or not safe_issubclass(
            self._schema._get_field_type(fields[0]), AnyDocArray  # type: ignore
        ):
            raise ValueError(f'subindex {subindex} is not valid')

        if len(fields) == 1:
            return self._subindices[fields[0]].find(
                query, search_field=search_field, limit=limit, **kwargs
            )

        return self._subindices[fields[0]]._find_subdocs(
            query,
            subindex='___'.join(fields[1:]),
            search_field=search_field,
            limit=limit,
            **kwargs,
        )

    def _get_root_doc_id(self, id: str, root: str, sub: str) -> str:
        """Get the root_id given the id of a subindex Document and the root and subindex name

        :param id: id of the subindex Document
        :param root: root index name
        :param sub: subindex name
        :return: the root_id of the Document
        """
        subindex = self._subindices[root]

        if not sub:
            sub_doc = subindex._get_items([id])
            parent_id = (
                sub_doc[0]['parent_id']
                if isinstance(sub_doc[0], dict)
                else sub_doc[0].parent_id
            )
            return parent_id
        else:
            fields = sub.split('__')
            cur_root_id = subindex._get_root_doc_id(
                id, fields[0], '__'.join(fields[1:])
            )
            return self._get_root_doc_id(cur_root_id, root, '')

    def subindex_contains(self, item: BaseDoc) -> bool:
        """Checks if a given BaseDoc item is contained in the index or any of its subindices.

        :param item: the given BaseDoc
        :return: if the given BaseDoc item is contained in the index/subindices
        """
        if self.num_docs() == 0:
            return False

        if safe_issubclass(type(item), BaseDoc):
            return self.__contains__(item) or any(
                index.subindex_contains(item) for index in self._subindices.values()
            )
        else:
            raise TypeError(
                f"item must be an instance of BaseDoc or its subclass, not '{type(item).__name__}'"
            )
