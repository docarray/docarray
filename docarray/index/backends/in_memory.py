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
from docarray.array.any_array import AnyDocArray
from docarray.helper import _shallow_copy_doc
from docarray.index.abstract import BaseDocIndex, _raise_not_supported
from docarray.index.backends.helper import (
    _collect_query_args,
    _execute_find_and_filter_query,
)
from docarray.typing import AnyTensor, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.filter import filter_docs
from docarray.utils.find import (
    FindResult,
    FindResultBatched,
    _da_attr_type,
    _extract_embeddings,
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
        if 'db_config' in kwargs:
            kwargs.pop('db_config')
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

                data_by_columns = self._get_col_value_dict(self._docs)
                self._update_subindex_data(self._docs)
                self._index_subindex(data_by_columns)

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

        self._embedding_map: Dict[str, Tuple[AnyTensor, Optional[List[int]]]] = {}

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        return python_type

    @property
    def out_schema(self) -> Type[BaseDoc]:
        """Return the original schema (without the parent_id from new_schema type)"""
        if self._is_subindex:
            return self._ori_schema
        return cast(Type[BaseDoc], self._schema)

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

        # Add parent_id to all sub-index documents and store sub-index documents
        data_by_columns = self._get_col_value_dict(docs)
        self._update_subindex_data(docs)
        self._index_subindex(data_by_columns)

        self._rebuild_embedding()

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        raise NotImplementedError

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        return len(self._docs)

    def _rebuild_embedding(self):
        """
        Reconstructs the embeddings map for each field. This is performed to store pre-stacked
        embeddings, thereby optimizing performance by avoiding repeated stacking of embeddings.

        Note: '_embedding_map' is a dictionary mapping fields to their corresponding embeddings.
        """
        if self.num_docs() == 0:
            self._embedding_map = dict()
        else:
            for field_, embedding in self._embedding_map.items():
                embedding_type = _da_attr_type(self._docs, field_)
                self._embedding_map[field_] = _extract_embeddings(
                    self._docs, field_, embedding_type
                )

    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        for field_, type_, _ in self._flatten_schema(cast(Type[BaseDoc], self._schema)):
            if safe_issubclass(type_, AnyDocArray):
                for id in doc_ids:
                    doc_ = self._get_items([id])
                    if len(doc_) == 0:
                        raise KeyError(
                            f"The document (id = '{id}') does not exist in the ExactNNIndexer."
                        )
                    sub_ids = [sub_doc.id for sub_doc in getattr(doc_[0], field_)]
                    del self._subindices[field_][sub_ids]

        indices = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                indices.append(i)

        del self._docs[indices]
        self._rebuild_embedding()

    def _ori_items(self, doc: BaseDoc) -> BaseDoc:
        """
        The Indexer's backend stores parent_id to support nested data. However,
        this method enables us to retrieve the original items in their original
        type, which is what the user interacts with.

        :param doc: The input document in New_Schema format from the Indexer's backend.
        :return: The input document with its original schema.
        """

        ori_doc = _shallow_copy_doc(doc)
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self.out_schema)
        ):
            if safe_issubclass(type_, AnyDocArray):
                _list = getattr(ori_doc, field_name)
                for i, nested_doc in enumerate(_list):
                    sub_indexer: InMemoryExactNNIndex = cast(
                        InMemoryExactNNIndex, self._subindices[field_name]
                    )
                    nested_doc = self._subindices[field_name]._ori_schema(
                        **nested_doc.__dict__
                    )

                    _list[i] = sub_indexer._ori_items(nested_doc)

        return ori_doc

    def _get_items(
        self, doc_ids: Sequence[str], raw: bool = False
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :param raw: if raw, output the new_schema type (with parent id)
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`.
            Duplicate `doc_ids` can be omitted in the output.
        """

        out_docs = []
        for i, doc in enumerate(self._docs):
            if doc.id in doc_ids:
                if raw:
                    out_docs.append(doc)
                else:
                    ori_doc = self._ori_items(doc)
                    schema_cls = cast(Type[BaseDoc], self.out_schema)
                    new_doc = schema_cls(**ori_doc.__dict__)
                    out_docs.append(new_doc)

        return out_docs

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
            cache=self._embedding_map,
        )

        docs_ = []
        for doc in docs:
            ori_doc = self._ori_items(doc)
            schema_cls = cast(Type[BaseDoc], self.out_schema)
            docs_.append(schema_cls(**ori_doc.__dict__))

        docs_with_schema = DocList.__class_getitem__(
            cast(Type[BaseDoc], self.out_schema)
        )(docs_)

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
            cache=self._embedding_map,
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

        docs = filter_docs(docs=self._docs, query=filter_query)[:limit]
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

    def _get_root_doc_id(self, id: str, root: str, sub: str) -> str:
        """Get the root_id given the id of a subindex Document and the root and subindex name

        :param id: id of the subindex Document
        :param root: root index name
        :param sub: subindex name
        :return: the root_id of the Document
        """
        subindex: InMemoryExactNNIndex = cast(
            InMemoryExactNNIndex, self._subindices[root]
        )

        if not sub:
            sub_doc = subindex._get_items([id], raw=True)
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
