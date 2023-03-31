from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, TypeVar, Union

import numpy as np

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.index.abstract import BaseDocIndex, _ColumnInfo
from docarray.typing import AnyTensor
from docarray.utils.find import _FindResult

TSchema = TypeVar('TSchema', bound=BaseDoc)
T = TypeVar('T', bound='ElasticV7DocIndex')


class ElasticV7DocIndex(ElasticDocIndex):

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################

    class QueryBuilder(ElasticDocIndex.QueryBuilder):
        def build(self, *args, **kwargs) -> Any:
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
        ):
            if isinstance(query, BaseDoc):
                query_vec = BaseDocIndex._get_values_by_column([query], search_field)[0]
            else:
                query_vec = query
            query_vec_np = BaseDocIndex._to_numpy(self._outer_instance, query_vec)
            self._query['size'] = limit
            self._query['query']['script_score'] = ElasticV7DocIndex._form_search_body(
                query_vec_np, limit, search_field
            )['query']['script_score']

            return self

    @dataclass
    class DBConfig(ElasticDocIndex.DBConfig):
        hosts: Union[str, List[str], None] = 'http://localhost:9200'  # type: ignore

    @dataclass
    class RuntimeConfig(ElasticDocIndex.RuntimeConfig):
        def dense_vector_config(self):
            return {'dims': 128}

    ###############################################
    # Implementation of abstract methods          #
    ###############################################

    def execute_query(self, query: Dict[str, Any], *args, **kwargs) -> Any:
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        resp = self._client.search(index=self._index_name, body=query)
        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=scores)

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

    @staticmethod
    def _form_search_body(query: np.ndarray, limit: int, search_field: str = '') -> Dict[str, Any]:  # type: ignore
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
        self._client.indices.put_mapping(index=self._index_name, body=mappings)

    def _client_create(self, mappings: Dict[str, Any]):
        body = {'mappings': mappings}
        self._client.indices.create(index=self._index_name, body=body)

    def _client_put_settings(self, settings: Dict[str, Any]):
        self._client.indices.put_settings(index=self._index_name, body=settings)

    def _client_mget(self, ids: Sequence[str]):
        return self._client.mget(index=self._index_name, body={'ids': ids})

    def _client_search(self, **kwargs):
        return self._client.search(index=self._index_name, body=kwargs)

    def _client_msearch(self, request: List[Dict[str, Any]]):
        return self._client.msearch(index=self._index_name, body=request)
