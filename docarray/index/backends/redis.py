import uuid
from typing import (
    TypeVar,
    Generic,
    Optional,
    List,
    Dict,
    Any,
    Sequence,
    Union,
    Generator,
    Type,
    cast,
    TYPE_CHECKING,
    Iterator,
    Mapping,
    Tuple,
)
from dataclasses import dataclass, field

import json
import numpy as np
from numpy import ndarray

from docarray.index.backends.helper import _collect_query_args
from docarray import BaseDoc, DocList
from docarray.index.abstract import (
    BaseDocIndex,
    _raise_not_composable,
)
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library
from docarray.utils.find import _FindResultBatched, _FindResult, FindResult

if TYPE_CHECKING:
    import redis
    from redis.commands.search.query import Query
    from redis.commands.search.field import (
        NumericField,
        TextField,
        VectorField,
    )
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
else:
    redis = import_library('redis')

    from redis.commands.search.field import (
        NumericField,
        TextField,
        VectorField,
    )
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

TSchema = TypeVar('TSchema', bound=BaseDoc)

VALID_DISTANCES = ['L2', 'IP', 'COSINE']
VALID_ALGORITHMS = ['FLAT', 'HNSW']
VALID_TEXT_SCORERS = [
    'BM25',
    'TFIDF',
    'TFIDF.DOCNORM',
    'DISMAX',
    'DOCSCORE',
    'HAMMING',
]


class RedisDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(RedisDocumentIndex.DBConfig, self._db_config)

        if not self._db_config.index_name:
            self._db_config.index_name = 'index_name__' + self._random_name()
        self._prefix = self._db_config.index_name + ':'

        # initialize Redis client
        self._client = redis.Redis(
            host=self._db_config.host,
            port=self._db_config.port,
            username=self._db_config.username,
            password=self._db_config.password,
            decode_responses=False,
        )
        self._create_index()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @staticmethod
    def _random_name():
        return uuid.uuid4().hex

    def _create_index(self):
        if not self._check_index_exists(self._db_config.index_name):
            schema = []
            for column, info in self._column_infos.items():
                if info.db_type == VectorField:
                    space = info.config.get('space')
                    if space:
                        for valid_dist in VALID_DISTANCES:
                            if space.upper() == valid_dist:
                                space = valid_dist
                    if space not in VALID_DISTANCES:
                        space = self._db_config.distance

                    attributes = {
                        'TYPE': 'FLOAT32',
                        'DIM': info.n_dim or info.config.get('dim'),
                        'DISTANCE_METRIC': space,
                        'EF_CONSTRUCTION': self._db_config.ef_construction,
                        'EF_RUNTIME': self._db_config.ef_runtime,
                        'M': self._db_config.m,
                        'INITIAL_CAP': self._db_config.initial_cap,
                    }
                    attributes = {
                        name: value for name, value in attributes.items() if value
                    }
                    schema.append(
                        info.db_type(
                            '$.' + column,
                            algorithm=info.config.get(
                                'algorithm', self._db_config.algorithm
                            ),
                            attributes=attributes,
                            as_name=column,
                        )
                    )
                else:
                    schema.append(info.db_type('$.' + column, as_name=column))

            # Create Redis Index
            self._client.ft(self._db_config.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self._prefix], index_type=IndexType.JSON
                ),
            )

            self._logger.info(f'index {self._db_config.index_name} has been created')
        else:
            self._logger.info(
                f'connected to existing {self._db_config.index_name} index'
            )

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            self._client.ft(index_name).info()
        except:  # noqa: E722
            self._logger.info("Index does not exist")
            return False
        self._logger.info("Index already exists")
        return True

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            return self._queries

        find = _collect_query_args('find')
        filter = _collect_query_args('filter')
        text_search = _raise_not_composable('text_search')
        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search_batched = _raise_not_composable('text_search_batched')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of RedisDocumentIndex."""

        host: str = 'localhost'
        port: int = 6379
        index_name: Optional[str] = None
        username: Optional[str] = None
        password: Optional[str] = None
        algorithm: str = field(default='FLAT')
        distance: str = field(default='COSINE')
        text_scorer: str = field(default='BM25')
        ef_construction: Optional[int] = None
        m: Optional[int] = None
        ef_runtime: Optional[int] = None
        block_size: Optional[int] = None
        initial_cap: Optional[int] = None

        def __post_init__(self):
            self.algorithm = self.algorithm.upper()
            self.distance = self.distance.upper()
            self.text_scorer = self.text_scorer.upper()
            if self.algorithm not in VALID_ALGORITHMS:
                raise ValueError(
                    f"Invalid algorithm '{self.algorithm}' provided. "
                    f"Must be one of: {', '.join(VALID_ALGORITHMS)}"
                )

            if self.distance not in VALID_DISTANCES:
                raise ValueError(
                    f"Invalid distance metric '{self.distance}' provided. "
                    f"Must be one of: {', '.join(VALID_DISTANCES)}"
                )

            if self.text_scorer not in VALID_TEXT_SCORERS:
                raise ValueError(
                    f"Invalid text scorer '{self.text_scorer}' provided. "
                    f"Must be one of: {', '.join(VALID_TEXT_SCORERS)}"
                )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of RedisDocumentIndex."""

        default_column_config: Dict[Any, Dict[str, Any]] = field(
            default_factory=lambda: {
                TextField: {},
                NumericField: {},
                VectorField: {},
            }
        )

    def python_type_to_db_type(self, python_type: Type) -> Any:
        type_map = {
            int: NumericField,
            float: NumericField,
            str: TextField,
            bytes: TextField,
            np.ndarray: VectorField,
            list: VectorField,
            AbstractTensor: VectorField,
        }

        for py_type, redis_type in type_map.items():
            if issubclass(python_type, py_type):
                return redis_type
        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    @staticmethod
    def _generate_item(
        column_to_data: Dict[str, Generator[Any, None, None]]
    ) -> Iterator[Dict[str, Any]]:
        """
        Given a dictionary of generators, yield a dictionary where each item consists of a key and
        a single item from the corresponding generator.

        :param column_to_data: A dictionary where each key is a column and each value
            is a generator.

        :yield: A dictionary where each item consists of a column name and an item from
            the corresponding generator. Yields until all generators are exhausted.
        """
        keys = list(column_to_data.keys())
        iterators = [iter(column_to_data[key]) for key in keys]
        while True:
            item_dict = {}
            for key, it in zip(keys, iterators):
                item = next(it, None)

                if key == 'id' and not item:
                    return

                if isinstance(item, AbstractTensor):
                    item_dict[key] = item._docarray_to_ndarray().tolist()
                elif isinstance(item, ndarray):
                    item_dict[key] = item.astype(np.float32).tolist()
                elif item is not None:
                    item_dict[key] = item

            yield item_dict

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        ids = []
        for item in self._generate_item(column_to_data):
            ids.append(item['id'])
            doc_id = self._prefix + item['id']
            self._client.json().set(doc_id, '$', item)

        num_docs = self.num_docs()
        print('indexed', num_docs)
        return ids

    def num_docs(self) -> int:
        num_docs = self._client.ft(self._db_config.index_name).info()['num_docs']
        return int(num_docs)

    def _del_items(self, doc_ids: Sequence[str]):
        doc_ids = [self._prefix + id for id in doc_ids if self._doc_exists(id)]
        if doc_ids:
            self._client.delete(*doc_ids)

    def _doc_exists(self, doc_id):
        return self._client.exists(self._prefix + doc_id)

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        if not doc_ids:
            return []

        docs = []
        for id in doc_ids:
            doc = self._client.json().get(self._prefix + id)
            if doc:
                docs.append(doc)

        if len(docs) == 0:
            raise KeyError(f'No document with id {doc_ids} found')
        return docs

    def execute_query(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        components: Dict[str, List[Dict[str, Any]]] = {}
        for component, value in query:
            if component not in components:
                components[component] = []
            components[component].append(value)

        if (
            len(components) != 2
            or len(components.get('find', [])) != 1
            or len(components.get('filter', [])) != 1
        ):
            raise ValueError(
                'The query must contain exactly one "find" and "filter" components.'
            )

        filter_query = components['filter'][0]['filter_query']
        query = components['find'][0]['query']
        search_field = components['find'][0]['search_field']
        limit = (
            components['find'][0].get('limit')
            or components['filter'][0].get('limit')
            or 10
        )
        docs, scores = self._hybrid_search(
            query=query,
            filter_query=filter_query,
            search_field=search_field,
            limit=limit,
        )
        docs = self._dict_list_to_docarray(docs)
        return FindResult(documents=docs, scores=scores)

    def _hybrid_search(
        self, query: np.ndarray, filter_query: str, search_field: str, limit: int
    ):
        redis_query = (
            Query(f'{filter_query}=>[KNN {limit} @{search_field} $vec AS vector_score]')
            .sort_by('vector_score')
            .paging(0, limit)
            .dialect(2)
        )
        query_params: Mapping[str, str] = {  # type: ignore
            'vec': np.array(query, dtype=np.float32).tobytes()  # type: ignore
        }
        results = (
            self._client.ft(self._db_config.index_name)
            .search(redis_query, query_params)
            .docs
        )

        scores: NdArray = NdArray._docarray_from_native(
            np.array([document['vector_score'] for document in results])
        )

        docs = []
        for out_doc in results:
            doc_dict = json.loads(out_doc.json)
            docs.append(doc_dict)
        return _FindResult(documents=docs, scores=scores)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        return self._hybrid_search(
            query=query, filter_query='*', search_field=search_field, limit=limit
        )

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        docs, scores = [], []
        for query in queries:
            results = self._find(query=query, search_field=search_field, limit=limit)
            docs.append(results.documents)
            scores.append(results.scores)

        return _FindResultBatched(documents=docs, scores=scores)

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        q = Query(filter_query)
        q.paging(0, limit)

        results = self._client.ft(index_name=self._db_config.index_name).search(q).docs
        docs = [json.loads(doc.json) for doc in results]
        return docs

    # def _build_query_node(self, key, condition):
    #     operator = list(condition.keys())[0]
    #     value = condition[operator]
    #
    #     query_dict = {}
    #
    #     if operator in ['$ne', '$eq']:
    #         if isinstance(value, bool):
    #             query_dict[key] = equal(int(value))
    #         elif isinstance(value, (int, float)):
    #             query_dict[key] = equal(value)
    #         else:
    #             query_dict[key] = '"' + value + '"'
    #     elif operator == '$gt':
    #         query_dict[key] = gt(value)
    #     elif operator == '$gte':
    #         query_dict[key] = ge(value)
    #     elif operator == '$lt':
    #         query_dict[key] = lt(value)
    #     elif operator == '$lte':
    #         query_dict[key] = le(value)
    #     else:
    #         raise ValueError(
    #             f'Expecting filter operator one of $gt, $gte, $lt, $lte, $eq, $ne, $and OR $or, got {operator} instead'
    #         )
    #
    #     if operator == '$ne':
    #         return DistjunctUnion(**query_dict)
    #     return IntersectNode(**query_dict)
    #
    # def _build_query_nodes(self, filter):
    #     nodes = []
    #     for k, v in filter.items():
    #         if k == '$and':
    #             children = self._build_query_nodes(v)
    #             node = intersect(*children)
    #             nodes.append(node)
    #         elif k == '$or':
    #             children = self._build_query_nodes(v)
    #             node = union(*children)
    #             nodes.append(node)
    #         else:
    #             child = self._build_query_node(k, v)
    #             nodes.append(child)
    #
    #     return nodes
    #
    # def _get_redis_filter_query(self, filter: Union[str, Dict]):
    #     if isinstance(filter, dict):
    #         nodes = self._build_query_nodes(filter)
    #         query_str = intersect(*nodes).to_string()
    #     elif isinstance(filter, str):
    #         query_str = filter
    #     else:
    #         raise ValueError(f'Unexpected type of filter: {type(filter)}, expected str')
    #
    #     return query_str

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        results = []
        for query in filter_queries:
            results.append(self._filter(filter_query=query, limit=limit))
        return results

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        query_str = '|'.join(query.split(' '))
        q = (
            Query(f'@{search_field}:{query_str}')
            .scorer(self._db_config.text_scorer)
            .with_scores()
            .paging(0, limit)
        )

        results = self._client.ft(index_name=self._db_config.index_name).search(q).docs

        scores: NdArray = NdArray._docarray_from_native(
            np.array([document['score'] for document in results])
        )

        docs = [json.loads(doc.json) for doc in results]

        return _FindResult(documents=docs, scores=scores)

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        docs, scores = [], []
        for query in queries:
            results = self._text_search(
                query=query, search_field=search_field, limit=limit
            )
            docs.append(results.documents)
            scores.append(results.scores)

        return _FindResultBatched(documents=docs, scores=scores)
