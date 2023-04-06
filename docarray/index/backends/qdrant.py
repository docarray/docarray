import uuid
from dataclasses import dataclass, field
from typing import (
    TypeVar,
    Generic,
    Optional,
    cast,
    Sequence,
    Any,
    Union,
    List,
    Dict,
    Generator,
    Type,
    Tuple,
)

import numpy as np
from grpc._channel import _InactiveRpcError
from qdrant_client.http.exceptions import UnexpectedResponse

import docarray.typing.id
from docarray import BaseDoc, DocList
from docarray.index.abstract import (
    BaseDocIndex,
    _FindResultBatched,
    _ColumnInfo,
    _raise_not_composable,
)

import qdrant_client
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models as rest

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import torch_imported
from docarray.utils.find import _FindResult

TSchema = TypeVar('TSchema', bound=BaseDoc)


QDRANT_PY_VEC_TYPES: List[Any] = [np.ndarray, AbstractTensor]
if torch_imported:
    import torch

    QDRANT_PY_VEC_TYPES.append(torch.Tensor)

QDRANT_SPACE_MAPPING = {
    'cosine': rest.Distance.COSINE,
    'l2': rest.Distance.EUCLID,
    'ip': rest.Distance.DOT,
}


class QdrantDocumentIndex(BaseDocIndex, Generic[TSchema]):
    UUID_NAMESPACE = uuid.UUID('3896d314-1e95-4a3a-b45a-945f9f0b541d')

    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(QdrantDocumentIndex.DBConfig, self._db_config)
        self._client = qdrant_client.QdrantClient(
            location=self._db_config.location,
            url=self._db_config.url,
            port=self._db_config.port,
            grpc_port=self._db_config.grpc_port,
            prefer_grpc=self._db_config.prefer_grpc,
            https=self._db_config.https,
            api_key=self._db_config.api_key,
            prefix=self._db_config.prefix,
            timeout=self._db_config.timeout,
            host=self._db_config.host,
            path=self._db_config.path,
        )
        self._initialize_collection()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @dataclass
    class Query:
        vector_field: Optional[str]
        vector_query: Optional[np.array]
        filter: Optional[rest.Filter]
        limit: int

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(
            self,
            vector_search_field: Optional[str] = None,
            vector_filters: Optional[List[np.array]] = None,
            payload_filters: Optional[List[rest.Filter]] = None,
            text_search_filters: Optional[List[Tuple[str, str]]] = None,
        ):
            self._vector_search_field: Optional[str] = vector_search_field
            self._vector_filters: List[np.array] = vector_filters or []
            self._payload_filters: List[rest.Filter] = payload_filters or []
            self._text_search_filters: List[Tuple[str, str]] = text_search_filters or []

        def build(self, limit: int) -> 'QdrantDocumentIndex.Query':
            vector_query = None
            if len(self._vector_filters) > 0:
                # If there are multiple vector queries applied, we can average them and
                # perform semantic search on a single vector instead
                vector_query = np.average(self._vector_filters, axis=0)
            merged_filter = None
            if len(self._payload_filters) > 0:
                merged_filter = rest.Filter(must=self._payload_filters)
            if len(self._text_search_filters) > 0:
                # Text search is just a special case of payload filtering, so the
                # payload filter is simply extended
                merged_filter = merged_filter or rest.Filter(must=[])
                for search_field, query in self._text_search_filters:
                    merged_filter.must.append(
                        rest.FieldCondition(
                            key=search_field,
                            match=rest.MatchText(text=query),
                        )
                    )
            return QdrantDocumentIndex.Query(
                vector_field=self._vector_search_field,
                vector_query=vector_query,
                filter=merged_filter,
                limit=limit
            )

        def find(
            self, query: np.ndarray, search_field: str = ''
        ) -> 'QdrantDocumentIndex.QueryBuilder':
            if self._vector_search_field and self._vector_search_field != search_field:
                raise ValueError(
                    f'Trying to call .find for search_field = {search_field}, but '
                    f'previously {self._vector_search_field} was used. Only a single '
                    f'field might be used in chained calls.'
                )
            return QdrantDocumentIndex.QueryBuilder(
                vector_search_field=search_field,
                vector_filters=self._vector_filters + [query],
                payload_filters=self._payload_filters,
                text_search_filters=self._text_search_filters,
            )

        def filter(
            self, filter_query: rest.Filter
        ) -> 'QdrantDocumentIndex.QueryBuilder':
            return QdrantDocumentIndex.QueryBuilder(
                vector_search_field=self._vector_search_field,
                vector_filters=self._vector_filters,
                payload_filters=self._payload_filters + [filter_query],
                text_search_filters=self._text_search_filters,
            )

        def text_search(
            self, query: str, search_field: str = ''
        ) -> 'QdrantDocumentIndex.QueryBuilder':
            return QdrantDocumentIndex.QueryBuilder(
                vector_search_field=self._vector_search_field,
                vector_filters=self._vector_filters,
                payload_filters=self._payload_filters,
                text_search_filters=self._text_search_filters + [(search_field, query)],
            )

        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('find_batched')
        text_search_batched = _raise_not_composable('text_search')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        location: Optional[str] = None
        url: Optional[str] = None
        port: Optional[int] = 6333
        grpc_port: int = 6334
        prefer_grpc: bool = True
        https: Optional[bool] = None
        api_key: Optional[str] = None
        prefix: Optional[str] = None
        timeout: Optional[float] = None
        host: Optional[str] = None
        path: Optional[str] = None
        collection_name: str = 'documents'
        shard_number: Optional[int] = None
        replication_factor: Optional[int] = None
        write_consistency_factor: Optional[int] = None
        on_disk_payload: Optional[bool] = None
        hnsw_config: Optional[types.HnswConfigDiff] = None
        optimizers_config: Optional[types.OptimizersConfigDiff] = None
        wal_config: Optional[types.WalConfigDiff] = None
        quantization_config: Optional[types.QuantizationConfig] = None

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                'id': {},
                'vector': {},
                'payload': {},
                np.ndarray: {},
            }
        )

    def python_type_to_db_type(self, python_type: Type) -> Any:
        for vector_type in QDRANT_PY_VEC_TYPES:
            if issubclass(python_type, vector_type):
                return 'vector'

        if issubclass(python_type, docarray.typing.id.ID):
            return 'id'

        return 'payload'

    def _initialize_collection(self):
        try:
            self._client.get_collection(self._db_config.collection_name)
            # TODO: handle different configuration of the collection
        except (UnexpectedResponse, _InactiveRpcError):
            vectors_config = {
                column_name: self._to_qdrant_vector_params(column_info)
                for column_name, column_info in self._column_infos.items()
                if column_info.db_type == 'vector'
            }
            self._client.create_collection(
                collection_name=self._db_config.collection_name,
                vectors_config=vectors_config,
                shard_number=self._db_config.shard_number,
                replication_factor=self._db_config.replication_factor,
                write_consistency_factor=self._db_config.write_consistency_factor,
                on_disk_payload=self._db_config.on_disk_payload,
                hnsw_config=self._db_config.hnsw_config,
                optimizers_config=self._db_config.optimizers_config,
                wal_config=self._db_config.wal_config,
                quantization_config=self._db_config.quantization_config,
            )

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        rows = self._transpose_col_value_dict(column_to_data)
        # TODO: add batching the documents to avoid timeouts
        points = [self._build_point_from_row(row) for row in rows]
        self._client.upsert(
            collection_name=self._db_config.collection_name,
            points=points,
        )

    def num_docs(self) -> int:
        return self._client.count(collection_name=self._db_config.collection_name).count

    def _del_items(self, doc_ids: Sequence[str]):
        items = self._get_items(doc_ids)
        if len(items) < len(doc_ids):
            found_keys = set(item['id'] for item in items)
            missing_keys = set(doc_ids) - found_keys
            raise KeyError('Document keys could not found: %s' % ','.join(missing_keys))

        self._client.delete(
            collection_name=self._db_config.collection_name,
            points_selector=rest.PointIdsList(
                points=[self._to_qdrant_id(doc_id) for doc_id in doc_ids],
            ),
        )

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        response, _ = self._client.scroll(
            collection_name=self._db_config.collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.HasIdCondition(
                        has_id=[self._to_qdrant_id(doc_id) for doc_id in doc_ids],
                    ),
                ],
            ),
            limit=len(doc_ids),
            with_payload=True,
            with_vectors=True,
        )
        return [self._convert_to_doc(point) for point in response]

    def execute_query(self, query: Query, *args, **kwargs) -> DocList:
        if query.vector_field:
            # We perform semantic search with some vectors with Qdrant's search metho
            # should be called
            points = self._client.search(
                collection_name=self._db_config.collection_name,
                query_vector=(query.vector_field, query.vector_query),
                query_filter=query.filter,
                limit=query.limit,
                with_payload=True,
                with_vectors=True,
            )
        else:
            # Just filtering, so Qdrant's scroll has to be used instead
            points, _ = self._client.scroll(
                collection_name=self._db_config.collection_name,
                scroll_filter=query.filter,
                limit=query.limit,
                with_payload=True,
                with_vectors=True,
            )

        docs = [self._convert_to_doc(point) for point in points]
        return self._dict_list_to_docarray(docs)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        query_batched = np.expand_dims(query, axis=0)
        docs, scores = self._find_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )
        return _FindResult(documents=docs[0], scores=scores[0])

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        responses = self._client.search_batch(
            collection_name=self._db_config.collection_name,
            requests=[
                rest.SearchRequest(
                    vector=rest.NamedVector(
                        name=search_field,
                        vector=query.tolist(),  # type: ignore
                    ),
                    limit=limit,
                    with_vector=True,
                    with_payload=True,
                )
                for query in queries
            ],
        )
        return _FindResultBatched(
            documents=[
                [self._convert_to_doc(point) for point in response]
                for response in responses
            ],
            scores=[
                np.array([point.score for point in response]) for response in responses
            ],
        )

    def _filter(
        self, filter_query: rest.Filter, limit: int
    ) -> Union[DocList, List[Dict]]:
        query_batched = [filter_query]
        docs = self._filter_batched(filter_queries=query_batched, limit=limit)
        return docs[0]

    def _filter_batched(
        self, filter_queries: Sequence[rest.Filter], limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        responses = []
        for filter_query in filter_queries:
            # There is no batch scroll available in Qdrant client yet, so we need to
            # perform the queries one by one. It will be changed in the future versions.
            response, _ = self._client.scroll(
                collection_name=self._db_config.collection_name,
                scroll_filter=filter_query,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            responses.append(response)

        return [
            [self._convert_to_doc(point) for point in response]
            for response in responses
        ]

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        query_batched = [query]
        docs, scores = self._text_search_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )
        return _FindResult(documents=docs[0], scores=scores[0])

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        filter_queries = [
            rest.Filter(
                must=[
                    rest.FieldCondition(
                        key=search_field,
                        match=rest.MatchText(text=query),
                    )
                ]
            )
            for query in queries
        ]
        documents_batched = self._filter_batched(
            filter_queries=filter_queries, limit=limit
        )

        # Qdrant does not return any scores if we just filter the objects, without using
        # semantic search over vectors. Thus, each document is scored with a value of 1
        return _FindResultBatched(
            documents=documents_batched,
            scores=[np.ones(len(docs)) for docs in documents_batched],
        )

    def _build_point_from_row(self, row: Dict[str, Any]) -> rest.PointStruct:
        point_id = self._to_qdrant_id(row.get('id'))
        vectors = {
            column_name: row.get(column_name).tolist()
            for column_name, column_info in self._column_infos.items()
            if column_info.db_type == 'vector'
        }
        payload = {
            column_name: row.get(column_name)
            for column_name, column_info in self._column_infos.items()
            if column_info.db_type in ['id', 'payload']
        }
        return rest.PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload,
        )

    def _to_qdrant_id(self, external_id: Optional[str]) -> str:
        if external_id is None:
            return uuid.uuid4().hex
        return uuid.uuid5(QdrantDocumentIndex.UUID_NAMESPACE, external_id).hex

    def _to_qdrant_vector_params(self, column_info: _ColumnInfo) -> rest.VectorParams:
        return rest.VectorParams(
            size=column_info.n_dim or column_info.config.get('dim'),
            distance=QDRANT_SPACE_MAPPING[column_info.config.get('space', 'cosine')],
        )

    def _convert_to_doc(
        self, point: Union[rest.ScoredPoint, rest.Record]
    ) -> Dict[str, Any]:
        doc = point.payload
        for vector_name, vector in point.vector.items():
            doc[vector_name] = vector
        return doc
