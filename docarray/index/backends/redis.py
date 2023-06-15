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
    TYPE_CHECKING, Iterator,
)
from dataclasses import dataclass, field

import numpy as np
import pickle

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


TSchema = TypeVar('TSchema', bound=BaseDoc)


class RedisDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(RedisDocumentIndex.DBConfig, self._db_config)

        if not self._db_config.index_name:
            self._db_config.index_name = 'index_name__' + 'random_name'  # todo
        self._prefix = self._db_config.index_name + ':'

        # initialize Redis client
        self._client = redis.Redis(
            host=self._db_config.host,
            port=self._db_config.port,
            username=self._db_config.username,
            password=self._db_config.password,
        )
        self._create_index()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    def _create_index(self):
        if not self._check_index_exists(self._db_config.index_name):
            schema = []
            for column, info in self._column_infos.items():
                if info.db_type == VectorField:
                    schema.append(
                        info.db_type(
                            name=column,
                            algorithm=info.config.get(
                                'algorithm', self._db_config.algorithm
                            ),
                            attributes={
                                'TYPE': 'FLOAT32',
                                'DIM': info.n_dim,
                                'DISTANCE_METRIC': 'COSINE',
                            },
                        )
                    )
                else:
                    schema.append(info.db_type(name=column))


            # Create Redis Index
            self._client.ft(self._db_config.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH),
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
        algorithm: str = 'FLAT'

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
    def _generate_item(column_to_data: Dict[str, Generator[Any, None, None]]) -> Iterator[Dict[str, Any]]:
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
                if item is None:  # If item is not None, add it to the dictionary
                    continue
                if isinstance(item, AbstractTensor):
                    item_dict[key] = pickle.dumps(item)
                else:
                    item_dict[key] = item

            if not item_dict:  # If item_dict is empty, break the loop
                break
            yield item_dict

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        ids = []
        pipeline = self._client.pipeline(transaction=False)
        batch_size = 10
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
        pass

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        pass

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        pass

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        pass

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        pass

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        pass

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        pass

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        pass

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        pass
