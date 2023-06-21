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
)
from dataclasses import dataclass, field

import binascii
import numpy as np

from redis.commands.search.query import Query

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library
from docarray.utils.find import _FindResultBatched, _FindResult

if TYPE_CHECKING:
    import redis
else:
    redis = import_library('redis')

    from redis.commands.search.field import (
        NumericField,
        TextField,
        VectorField,
    )
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.querystring import (
        DistjunctUnion,
        IntersectNode,
        equal,
        ge,
        gt,
        intersect,
        le,
        lt,
        union,
    )

TSchema = TypeVar('TSchema', bound=BaseDoc)

VALID_DISTANCES = ['L2', 'IP', 'COSINE']
VALID_ALGORITHMS = ['FLAT', 'HNSW']


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

                    schema.append(
                        info.db_type(
                            name=column,
                            algorithm=info.config.get(
                                'algorithm', self._db_config.algorithm
                            ),
                            attributes={
                                'TYPE': 'FLOAT32',
                                'DIM': info.n_dim,
                                'DISTANCE_METRIC': space,
                            },
                        )
                    )
                else:
                    schema.append(info.db_type(name=column))

            # Create Redis Index
            self._client.ft(self._db_config.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(
                    prefix=[self._prefix], index_type=IndexType.HASH
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
        ef_construction: Optional[int] = None
        m: Optional[int] = None
        ef_runtime: Optional[int] = None
        block_size: Optional[int] = None
        initial_cap: Optional[int] = None

        def __post_init__(self):
            if self.algorithm not in VALID_ALGORITHMS:
                raise ValueError(f"Invalid algorithm '{self.algorithm}' provided. "
                                 f"Must be one of: {', '.join(VALID_ALGORITHMS)}")

            if self.distance not in VALID_DISTANCES:
                raise ValueError(f"Invalid distance metric '{self.distance}' provided. "
                                 f"Must be one of: {', '.join(VALID_DISTANCES)}")

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

                if item is None:
                    item_dict[key] = '__None__'
                elif isinstance(item, AbstractTensor):
                    item_dict[key] = np.array(
                        item._docarray_to_ndarray(), dtype=np.float32
                    ).tobytes()
                else:
                    item_dict[key] = item

            yield item_dict

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        ids = []
        pipeline = self._client.pipeline(transaction=False)
        batch_size = 10  # variable [1k]
        for item in self._generate_item(column_to_data):
            doc_id = self._prefix + item.pop('id')
            pipeline.hset(
                doc_id,
                mapping=item,
            )
            ids.append(doc_id)

            if len(ids) % batch_size == 0:
                pipeline.execute()

        pipeline.execute()

        num_docs = self.num_docs()
        print('indexed', num_docs)
        return ids

    def num_docs(self) -> int:
        return self._client.ft(self._db_config.index_name).info()['num_docs']

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

        pipe = self._client.pipeline()
        for id in doc_ids:
            pipe.hgetall(self._prefix + id)

        results = pipe.execute()

        docs = [
            {k.decode('utf-8'): v.decode('utf-8', 'ignore') for k, v in d.items()}
            for d in results
        ]

        docs = [{k: v for k, v in d.items() if k != 'tens'} for d in docs]  # todo (vector decoding problem)
        docs = [{k: None if v == '__None__' else v for k, v in d.items()} for d in docs]  # todo (converting to None)
        return docs

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        pass

    def _convert_to_schema(self, document):
        doc_kwargs = {}
        for column, info in self._column_infos.items():
            if column == 'id':
                doc_kwargs['id'] = document.id[len(self._prefix) :]
            elif document[column] == '__None__':
                doc_kwargs[column] = None
            elif info.db_type == VectorField:
                # byte_string = document[column]
                # byte_data = byte_string.encode('utf-8')
                doc_kwargs[column] = np.frombuffer(document[column], dtype=np.float32)
            elif info.db_type == NumericField:
                doc_kwargs[column] = info.docarray_type(document[column])
            else:
                doc_kwargs[column] = document[column]

        return doc_kwargs

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        limit = 5
        query_str = '*'
        redis_query = (
            Query(f'{query_str}=>[KNN {limit} @{search_field} $vec AS vector_score]')
            .sort_by('vector_score')
            .paging(0, limit)
            .dialect(2)
        )
        query_params: Mapping[str, str] = {  # type: ignore
            'vec': np.array(query, dtype=np.float32).tobytes()
        }
        results = (
            self._client.ft(self._db_config.index_name)
            .search(redis_query, query_params)
            .docs
        )

        scores = [document['vector_score'] for document in results]
        docs = [self._convert_to_schema(document) for document in results]

        return _FindResult(documents=docs, scores=scores)

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
        query_str = self._get_redis_filter_query(filter_query)
        q = Query(query_str)
        q.paging(0, limit)

        results = self._client.ft(index_name=self._db_config.index_name).search(q).docs
        docs = [self._convert_to_schema(document) for document in results]

        return docs

    def _build_query_node(self, key, condition):
        operator = list(condition.keys())[0]
        value = condition[operator]

        query_dict = {}

        if operator in ['$ne', '$eq']:
            if isinstance(value, bool):
                query_dict[key] = equal(int(value))
            elif isinstance(value, (int, float)):
                query_dict[key] = equal(value)
            else:
                query_dict[key] = value
        elif operator == '$gt':
            query_dict[key] = gt(value)
        elif operator == '$gte':
            query_dict[key] = ge(value)
        elif operator == '$lt':
            query_dict[key] = lt(value)
        elif operator == '$lte':
            query_dict[key] = le(value)
        else:
            raise ValueError(
                f'Expecting filter operator one of $gt, $gte, $lt, $lte, $eq, $ne, $and OR $or, got {operator} instead'
            )

        if operator == '$ne':
            return DistjunctUnion(**query_dict)
        return IntersectNode(**query_dict)

    def _build_query_nodes(self, filter):
        nodes = []
        for k, v in filter.items():
            if k == '$and':
                children = self._build_query_nodes(v)
                node = intersect(*children)
                nodes.append(node)
            elif k == '$or':
                children = self._build_query_nodes(v)
                node = union(*children)
                nodes.append(node)
            else:
                child = self._build_query_node(k, v)
                nodes.append(child)

        return nodes

    def _get_redis_filter_query(self, filter: Union[str, Dict]):
        if isinstance(filter, dict):
            nodes = self._build_query_nodes(filter)
            query_str = intersect(*nodes).to_string()
        elif isinstance(filter, str):
            query_str = filter
        else:
            raise ValueError(f'Unexpected type of filter: {type(filter)}, expected str')

        return query_str

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

        scorer = 'BM25'
        if scorer not in [
            'BM25',
            'TFIDF',
            'TFIDF.DOCNORM',
            'DISMAX',
            'DOCSCORE',
            'HAMMING',
        ]:
            raise ValueError(
                f'Expecting a valid text similarity ranking algorithm, got {scorer} instead'
            )
        q = (
            Query(f'@{search_field}:{query_str}')
            .scorer(scorer)
            .with_scores()
            .paging(0, limit)
        )

        results = self._client.ft(index_name=self._db_config.index_name).search(q).docs

        scores = [document['score'] for document in results]
        docs = [self._convert_to_schema(document) for document in results]

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
