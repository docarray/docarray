from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from pymilvus import Collection, DataType, connections, utility

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.typing import AnyTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import _FindResult, _FindResultBatched

TSchema = TypeVar('TSchema', bound=BaseDoc)


class MilvusDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config, **kwargs)
        self._db_config: MilvusDocumentIndex.DBConfig = cast(
            MilvusDocumentIndex.DBConfig, self._db_config
        )

        self._client = connections.connect(
            db_name="default",
            host=self._db_config.host,
            user=self._db_config.user,
            password=self._db_config.password,
            token=self._db_config.token,
        )

        self._connection = self._init_index()

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        host: str = "localhost"
        port: int = 19530
        user: Optional[str] = ""
        password: Optional[str] = ""
        token: Optional[str] = ""
        index_name = None

    def python_type_to_db_type(self, python_type: Type) -> Any:
        type_map = {
            int: DataType.INT32,
            float: DataType.FLOAT,
            str: DataType.STRING,
            bytes: DataType.STRING,
            np.ndarray: DataType.FLOAT_VECTOR,
            list: DataType.FLOAT_VECTOR,
            AnyTensor: DataType.FLOAT_VECTOR,
        }

        for py_type, db_type in type_map.items(type_map):
            if safe_issubclass(python_type, py_type):
                return db_type

        raise ValueError("No corresponding milvus type")

    def _init_index(self) -> Collection:
        # TODO: Loading schema into Milvus Collection
        if not utility.has_collection("docs"):
            # INIT schema
            pass
        return Collection("docs").load()

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
        pass

    def num_docs(self) -> int:
        return self._connection.num_entities

    def _del_items(self, doc_ids: Sequence[str]):
        pass

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        pass

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        pass

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        pass

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        pass

    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocList:
        pass

    def filter_batched(
        self,
        filter_queries: Any,
        limit: int = 10,
        **kwargs,
    ) -> List[DocList]:
        pass
