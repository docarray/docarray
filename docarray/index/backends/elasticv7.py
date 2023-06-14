import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.index.abstract import BaseDocIndex, _ColumnInfo
from docarray.typing import AnyTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils.find import _FindResult

TSchema = TypeVar('TSchema', bound=BaseDoc)
T = TypeVar('T', bound='ElasticV7DocIndex')


class ElasticV7DocIndex(ElasticDocIndex):
    def __init__(self, db_config=None, **kwargs):
        """Initialize ElasticV7DocIndex"""
        from elasticsearch import __version__ as __es__version__

        if __es__version__[0] > 7:
            raise ImportError(
                'ElasticV7DocIndex requires the elasticsearch library to be version 7.10.1'
            )

        super().__init__(db_config, **kwargs)

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################

    class QueryBuilder(ElasticDocIndex.QueryBuilder):
        def build(self, *args, **kwargs) -> Any:
            """Build the elastic search v7 query object."""
            if (
                'script_score' in self._query['query']
                and 'bool' in self._query['query']
                and len(self._query['query']['bool']) > 0
            ):
                self._query['query']['script_score']['query'] = {}
                self._query['query']['script_score']['query']['bool'] = self._query[
                    'query'
                ]['bool']
                del self._query['query']['bool']

            return self._query

        def find(
            self,
            query: Union[AnyTensor, BaseDoc],
            search_field: str = 'embedding',
            limit: int = 10,
            num_candidates: Optional[int] = None,
        ):
            """
            Find k-nearest neighbors of the query.

            :param query: query vector for KNN/ANN search. Has single axis.
            :param search_field: name of the field to search on
            :param limit: maximum number of documents to return per query
            :return: self
            """
            if num_candidates:
                warnings.warn('`num_candidates` is not supported in ElasticV7DocIndex')

            if isinstance(query, BaseDoc):
                query_vec = BaseDocIndex._get_values_by_column([query], search_field)[0]
            else:
                query_vec = query
            query_vec_np = BaseDocIndex._to_numpy(self._outer_instance, query_vec)
            self._query['size'] = limit
            self._query['query'][
                'script_score'
            ] = self._outer_instance._form_search_body(
                query_vec_np, limit, search_field
            )[
                'query'
            ][
                'script_score'
            ]

            return self

    @dataclass
    class DBConfig(ElasticDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of ElasticDocIndex."""

        hosts: Union[str, List[str], None] = 'http://localhost:9200'  # type: ignore

        def dense_vector_config(self):
            return {'dims': 128}

    @dataclass
    class RuntimeConfig(ElasticDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of ElasticDocIndex."""

        pass

    ###############################################
    # Implementation of abstract methods          #
    ###############################################

    def execute_query(self, query: Dict[str, Any], *args, **kwargs) -> Any:
        """
        Execute a query on the ElasticDocIndex.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :return: the result of the query
        """
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        resp = self._client.search(index=self.index_name, body=query)
        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=parse_obj_as(NdArray, scores))

    ###############################################
    # Helpers                                     #
    ###############################################

    # ElasticSearch helpers
    def _create_index_mapping(self, col: '_ColumnInfo') -> Dict[str, Any]:
        """Create a new HNSW index for a column, and initialize it."""

        index = col.config.copy()
        if 'type' not in index:
            index['type'] = col.db_type

        if col.db_type == 'dense_vector' and col.n_dim:
            index['dims'] = col.n_dim

        return index

    def _form_search_body(self, query: np.ndarray, limit: int, search_field: str = '') -> Dict[str, Any]:  # type: ignore
        body = {
            'size': limit,
            'query': {
                'script_score': {
                    'query': {'match_all': {}},
                    'script': {
                        'source': f'cosineSimilarity(params.query_vector, \'{search_field}\') + 1.0',
                        'params': {'query_vector': query},
                    },
                }
            },
        }
        return body

    ###############################################
    # API Wrappers                                #
    ###############################################

    def _client_put_mapping(self, mappings: Dict[str, Any]):
        self._client.indices.put_mapping(index=self.index_name, body=mappings)

    def _client_create(self, mappings: Dict[str, Any]):
        body = {'mappings': mappings}
        self._client.indices.create(index=self.index_name, body=body)

    def _client_put_settings(self, settings: Dict[str, Any]):
        self._client.indices.put_settings(index=self.index_name, body=settings)

    def _client_mget(self, ids: Sequence[str]):
        return self._client.mget(index=self.index_name, body={'ids': ids})

    def _client_search(self, **kwargs):
        return self._client.search(index=self.index_name, body=kwargs)

    def _client_msearch(self, request: List[Dict[str, Any]]):
        return self._client.msearch(index=self.index_name, body=request)
