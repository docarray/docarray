from typing import TypeVar, Generic, Optional, List, Dict, Any, Sequence, Union, Generator, Type
from dataclasses import dataclass, field

import numpy as np

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.find import _FindResultBatched, _FindResult
from redis.commands.search.field import NumericField, TextField, VectorField, GeoField


TSchema = TypeVar('TSchema', bound=BaseDoc)


class RedisDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of RedisDocumentIndex."""

        host: str = 'http://localhost:6379'
        index_name: Optional[str] = None
        username: Optional[str] = None
        password: Optional[str] = None

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
            np.ndarray: VectorField,
            list: VectorField,
            AbstractTensor: VectorField,
        }

        for py_type, redis_type in type_map.items():
            if issubclass(python_type, py_type):
                return redis_type
        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        pass

    def num_docs(self) -> int:
        pass

    def _del_items(self, doc_ids: Sequence[str]):
        pass

    def _get_items(self, doc_ids: Sequence[str]) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        pass

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        pass

    def _find(self, query: np.ndarray, limit: int, search_field: str = '') -> _FindResult:
        pass

    def _find_batched(self, queries: np.ndarray, limit: int, search_field: str = '') -> _FindResultBatched:
        pass

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        pass

    def _filter_batched(self, filter_queries: Any, limit: int) -> Union[List[DocList], List[List[Dict]]]:
        pass

    def _text_search(self, query: str, limit: int, search_field: str = '') -> _FindResult:
        pass

    def _text_search_batched(self, queries: Sequence[str], limit: int, search_field: str = '') -> _FindResultBatched:
        pass

