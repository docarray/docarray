import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.typing import AnyTensor
from docarray.typing.id import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.misc import (
    import_library,
    is_tf_available,
    is_torch_available,
)
from docarray.utils.find import _FindResult, _FindResultBatched

torch_available, tf_available = is_torch_available(), is_tf_available()

if torch_available:
    import torch

if tf_available:
    import tensorflow as tf  # type: ignore

if TYPE_CHECKING:
    import numpy as np
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
else:
    hnswlib = import_library('hnswlib', raise_error=True)
    np = import_library('numpy', raise_error=False)
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

ID_VARCHAR_LEN = 1024
SERIALIZED_VARCHAR_LEN = 65_535  # Maximum length that Milvus allows for a VARCHAR field

TSchema = TypeVar('TSchema', bound=BaseDoc)


class MilvusDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, index_name=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
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

        self._db_config.index_name = index_name
        self._validate_columns()
        self._create_collection_name()
        self._collection = self._init_index()
        self._build_index()
        self._logger.info(f"{self.__class__.__name__} has been initialized")

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        collection_name: str = None
        collection_description: str = ""
        host: str = "localhost"
        port: int = 19530
        user: Optional[str] = ""
        password: Optional[str] = ""
        token: Optional[str] = ""
        index_name: str = None
        index_type: str = "IVF_FLAT"
        index_metric: str = "L2"
        index_params: Dict = field(default_factory=lambda: {"nlist": 1024})
        search_params: Dict = field(
            default_factory=lambda: {
                "metric_type": "L2",
                "params": {"nprobe": 10},
                "offset": 5,
            }
        )
        serialize_config: Dict = field(default_factory=lambda: {"protocol": "protobuf"})
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(dict)
        )

    def python_type_to_db_type(self, python_type: Type) -> Any:
        type_map = {
            int: DataType.INT64,
            float: DataType.FLOAT,
            str: DataType.VARCHAR,
            bytes: DataType.VARCHAR,
            np.ndarray: DataType.FLOAT_VECTOR,
            list: DataType.FLOAT_VECTOR,
            AnyTensor: DataType.FLOAT_VECTOR,
            AbstractTensor: DataType.FLOAT_VECTOR,
        }

        if issubclass(python_type, ID):
            return DataType.INT64  # Primary key must be int64

        for py_type, db_type in type_map.items():
            if safe_issubclass(python_type, py_type):
                return db_type

        raise ValueError("No corresponding milvus type")

    def _init_index(self) -> Collection:
        if not utility.has_collection(self._db_config.collection_name):
            fields = [
                FieldSchema(
                    name="serialized",
                    dtype=DataType.VARCHAR,
                    max_length=SERIALIZED_VARCHAR_LEN,
                ),
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=256,
                ),
            ]  # Initiliaze the id
            fields.extend(
                [
                    FieldSchema(
                        name=column_name,
                        dtype=info.db_type,
                        is_primary=False,
                        **(
                            {'max_length': 256}
                            if info.db_type == DataType.VARCHAR
                            else {}
                        ),
                        **(
                            {'dim': info.n_dim}
                            if info.db_type == DataType.FLOAT_VECTOR
                            else {}
                        ),
                    )
                    for column_name, info in self._column_infos.items()
                    if column_name != "id"
                ]
            )
            self._logger.info("Collection has been created")
            return Collection(
                name=self._db_config.collection_name,
                schema=CollectionSchema(
                    fields=fields,
                    description=self._db_config.collection_description,
                ),
                using='default',
            )

        return Collection(self._db_config.collection_name)

    def _create_collection_name(self):
        if self._db_config.collection_name is None:
            id = uuid.uuid4().hex
            self._db_config.collection_name = f"{self.__class__.__name__}__" + id

        self._db_config.collection_name = ''.join(
            re.findall('[a-zA-Z0-9_]', self._db_config.collection_name)
        )

    def _validate_columns(self):
        """
        Validates if the DataFrame contains at least one vector column
        (Milvus' requirement) and checks that each vector column has
        dimension information specified.
        """
        vector_columns_exist = any(
            safe_issubclass(info.docarray_type, AbstractTensor)
            for info in self._column_infos.values()
        )

        if not vector_columns_exist:
            raise ValueError(
                "No vector columns found. Ensure that at least one column is of a vector type"
            )

        for column_name, info in self._column_infos.items():
            if info.n_dim is None and safe_issubclass(
                info.docarray_type, AbstractTensor
            ):
                raise ValueError(
                    f"Dimension information is missing for column '{column_name}' which is vector type"
                )

    def _build_index(self):
        index = {
            "index_type": self._db_config.index_type,
            "metric_type": self._db_config.index_metric,
            "params": self._db_config.index_params,
        }

        self._collection.create_index(self._db_config.index_name, index)
        self._logger.info(
            f"Index '{self._db_config.index_name}' has been successfully created"
        )

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
        """Index Documents into the index.

        !!! note
            Passing a sequence of Documents that is not a DocList
            (such as a List of Docs) comes at a performance penalty.
            This is because the Index needs to check compatibility between itself and
            the data. With a DocList as input this is a single check; for other inputs
            compatibility needs to be checked for every Document individually.

        :param docs: Documents to index.
        """

        docs = self._validate_docs(docs)
        entities = [[] for _ in range(len(self._column_infos.items()) + 1)]

        for i in range(len(docs)):
            entities[0].append(docs[i].to_base64(**self._db_config.serialize_config))
            for j, (column_name, info) in enumerate(self._column_infos.items()):
                column_value = docs[i].__getattr__(column_name)
                if isinstance(column_value, (tf.Tensor, torch.Tensor, np.ndarray)):
                    column_value = self._convert_to_vector(column_value)
                entities[j + 1].append(column_value)

        self._collection.insert(entities)
        self._collection.flush()
        self._logger.info(f"{len(docs)} documents has been indexed")

    def _convert_to_vector(self, column_value: AbstractTensor) -> Sequence[float]:
        """
        Converts a column value to a float vector.

        The database can only store float vectors, so this method is used to convert
        TensorFlow or PyTorch tensors to a format compatible with the database.

        :param column_value: The column value to be converted.
        :return: The converted float vector.
        """
        if type(column_value) is list:
            return column_value

        if len(column_value.shape) != 1:
            raise ValueError(
                'Unsupported: Milvus backend only supports one-dimensional vectors'
            )

        if isinstance(column_value, np.ndarray):
            return column_value.astype(float).tolist()
        elif torch_available and torch.is_tensor(column_value):
            return column_value.float().numpy().tolist()
        elif tf_available and tf.is_tensor(column_value):
            return column_value.numpy().astype(float).tolist()

    def num_docs(self) -> int:
        return self._collection.num_entities

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        self._collection.load()
        ret = self._collection.query(
            expr="id in " + str([id for id in doc_ids]),
            offset=0,
            limit=self.num_docs(),
            output_fields=["serialized"],
        )

        return [
            self._schema.from_base64(
                ret[i]["serialized"], **self._db_config.serialize_config
            )
            for i in range(len(ret))
        ]

    def _del_items(self, doc_ids: Sequence[str]):
        self._collection.delete(expr="id in " + str([id for id in doc_ids]))
        self._logger.info(f"{len(doc_ids)} documents has been deleted")

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> Union[DocList, List[Dict]]:
        ...

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> Union[List[DocList], List[List[Dict]]]:
        ...

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        raise NotImplementedError

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        ...

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        ...

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        self._collection.load()

        results = self._collection.search(
            data=self._convert_to_vector(query),
            anns_field=search_field,
            param=self._db_config.search_params,
            limit=limit,
            expr=None,
            output_fields=["serialized"],
            consistency_level="Strong",
        )

        self._collection.release()
        results = next(iter(results), None)  # Only consider the first element

        return _FindResult(
            documents=DocList[self._schema](
                [
                    self._schema.from_base64(
                        hit.entity.get('serialized'), **self._db_config.serialize_config
                    )
                    for hit in results
                ]
            ),
            scores=[hit.score for hit in results],
        )

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        ...

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        ...
