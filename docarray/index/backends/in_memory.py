import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex, _raise_not_supported
from docarray.index.backends.helper import (
    _collect_query_args,
    _execute_find_and_filter_query,
)
from docarray.typing import AnyTensor, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.filter import filter_docs
from docarray.utils.find import (
    FindResult,
    FindResultBatched,
    _FindResult,
    _FindResultBatched,
    find,
    find_batched,
)

TSchema = TypeVar('TSchema', bound=BaseDoc)


class InMemoryExactNNIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(
        self,
        docs: Optional[DocList] = None,
        index_file_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize InMemoryExactNNIndex"""
        super().__init__(db_config=None, **kwargs)
        self._runtime_config = self.RuntimeConfig()

        if docs and index_file_path:
            raise ValueError(
                'Initialize `InMemoryExactNNIndex` with either `docs` or '
                '`index_file_path`, not both. Provide `docs` for a fresh index, or '
                '`index_file_path` to use an existing file.'
            )

        if index_file_path:
            if os.path.exists(index_file_path):
                self._logger.info(
                    f'Loading index from a binary file: {index_file_path}'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                ).load_binary(file=index_file_path)
            else:
                self._logger.warning(
                    f'Index file does not exist: {index_file_path}. '
                    f'Initializing empty InMemoryExactNNIndex.'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                )()
        else:
            if docs:
                self._logger.info('Docs provided. Initializing with provided docs.')
                self._docs = docs
            else:
                self._logger.info(
                    'No docs or index file provided. Initializing empty InMemoryExactNNIndex.'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                )()

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        return python_type

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            return self._queries

        find = _collect_query_args('find')
        find_batched = _collect_query_args('find_batched')
        filter = _collect_query_args('filter')
        filter_batched = _raise_not_supported('find_batched')
        text_search = _raise_not_supported('text_search')
        text_search_batched = _raise_not_supported('text_search')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of InMemoryExactNNIndex."""

        pass

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of InMemoryExactNNIndex."""

        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(
                dict,
                {
                    AbstractTensor: {'space': 'cosine_sim'},
                },
            )
        )

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
        # implementing the public option because conversion to column dict is not needed
        docs = self._validate_docs(docs)
        self._docs.extend(docs)

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        raise NotImplementedError

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        return len(self._docs)

    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        indices = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                indices.append(i)

        del self._docs[indices]

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`.
            Duplicate `doc_ids` can be omitted in the output.
        """
        indices = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                indices.append(i)
        return self._docs[indices]

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        """
        Execute a query on the InMemoryExactNNIndex.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )
        find_res = _execute_find_and_filter_query(
            doc_index=self,
            query=query,
        )
        return find_res

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find Documents in the index using nearest-neighbor search.

        :param query: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.)
            with a single axis, or a Document
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of Documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `find` for search field {search_field}')
        self._validate_search_field(search_field)

        if self.num_docs() == 0:
            return FindResult(documents=[], scores=[])  # type: ignore

        config = self._column_infos[search_field].config

        docs, scores = find(
            index=self._docs,
            query=query,
            search_field=search_field,
            limit=limit,
            metric=config['space'],
        )
        docs_with_schema = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))(
            docs
        )
        return FindResult(documents=docs_with_schema, scores=scores)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        raise NotImplementedError

    def find_batched(
        self,
        queries: Union[AnyTensor, DocList],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find Documents in the index using nearest-neighbor search.

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
        self._validate_search_field(search_field)

        if self.num_docs() == 0:
            return FindResultBatched(documents=[], scores=[])  # type: ignore

        config = self._column_infos[search_field].config

        find_res = find_batched(
            index=self._docs,
            query=cast(NdArray, queries),
            search_field=search_field,
            limit=limit,
            metric=config['space'],
        )

        return find_res

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        raise NotImplementedError

    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocList:
        """Find documents in the index based on a filter query

        :param filter_query: the filter query to execute following the query
            language of
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        self._logger.debug(f'Executing `filter` for the query {filter_query}')

        docs = filter_docs(docs=self._docs, query=filter_query)
        return cast(DocList, docs)

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        raise NotImplementedError

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        raise NotImplementedError(f'{type(self)} does not support filtering.')

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def persist(self, file: str = 'in_memory_index.bin') -> None:
        """Persist InMemoryExactNNIndex into a binary file."""
        self._docs.save_binary(file=file)
