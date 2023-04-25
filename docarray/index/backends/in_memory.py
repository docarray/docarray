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
from docarray.index.backends.hnswlib import _collect_query_args
from docarray.typing import ID, AnyTensor, NdArray
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


class InMemoryDocIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, docs: Optional[DocList] = None, **kwargs):
        super().__init__(db_config=None, **kwargs)

        self._docs: DocList
        if docs is None:
            self._docs = DocList[self._schema]()
        else:
            self._docs = docs

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
        """Dataclass that contains all "static" configurations of InMemoryDocIndex."""

        pass

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of InMemoryDocIndex."""

        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {},
                str: {},
                int: {},
                float: {},
                list: {},
                set: {},
                dict: {},
                ID: {},
                AbstractTensor: {},
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
        )

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
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
        indices = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                indices.append(i)

        for idx in reversed(indices):
            self._docs.pop(idx)

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        indices = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                indices.append(i)
        return self._docs[indices]

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        """
        Execute a query on the InMemoryDocIndex.

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

        ann_docs = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))([])
        filter_conditions = []
        doc_to_score: Dict[BaseDoc, Any] = {}
        for op, op_kwargs in query:
            if op == 'find':
                docs, scores = self.find(**op_kwargs)
                ann_docs.extend(docs)
                doc_to_score.update(zip(docs.__getattribute__('id'), scores))
            elif op == 'filter':
                filter_conditions.append(op_kwargs['filter_query'])

        self._logger.debug(f'Executing query {query}')
        docs_filtered = ann_docs
        for cond in filter_conditions:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
            docs_filtered = docs_cls(filter_docs(docs_filtered, cond))

        self._logger.debug(f'{len(docs_filtered)} results found')
        docs_and_scores = zip(
            docs_filtered, (doc_to_score[doc.id] for doc in docs_filtered)
        )
        docs_sorted = sorted(docs_and_scores, key=lambda x: x[1])
        out_docs, out_scores = zip(*docs_sorted)
        return FindResult(documents=out_docs, scores=out_scores)

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:

        self._logger.debug(f'Executing `find` for search field {search_field}')
        self._validate_search_field(search_field)

        docs, scores = find(
            index=self._docs,
            query=query,
            search_field=search_field,
            limit=limit,
        )
        return FindResult(documents=DocList[self._schema](docs), scores=scores)

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
        self._validate_search_field(search_field)

        find_res = find_batched(
            index=self._docs,
            query=cast(NdArray, queries),
            search_field=search_field,
            limit=limit,
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

        :param filter_query: the DB specific filter query to execute
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
