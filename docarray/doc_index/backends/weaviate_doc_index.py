from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import weaviate

import docarray
from docarray import BaseDocument, DocumentArray
from docarray.doc_index.abstract_doc_index import BaseDocumentIndex, _FindResultBatched
from docarray.utils.find import FindResult, _FindResult

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='WeaviateDocumentIndex')


DEFAULT_BATCH_CONFIG = {
    "batch_size": 20,
    "dynamic": False,
    "timeout_retries": 3,
    "num_workers": 1,
}

# TODO: add more types
# see https://weaviate.io/developers/weaviate/configuration/datatypes
WEAVIATE_PY_VEC_TYPES = [list, tuple, np.ndarray]
WEAVIATE_PY_TYPES = [bool, int, float, str, docarray.typing.ID]


class WeaviateDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs) -> None:
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(WeaviateDocumentIndex.DBConfig, self._db_config)

        self._client = weaviate.Client(self._db_config.host)
        self._configure_client()
        self._create_schema()

    def _configure_client(self):
        self._client.batch.configure(**self._db_config.batch_config)

    def _create_schema(self):
        schema = {}

        properties = []
        column_infos = self._column_infos

        for column_name, column_info in column_infos.items():
            # in weaviate, we do not create a property for the vector
            if column_info.db_type == np.ndarray:
                continue
            prop = {
                "name": column_name
                if column_name != 'id'
                else '__id',  # in weaviate, id and _id is a reserved keyword
                "dataType": column_info.config["dataType"],
            }
            properties.append(prop)

        # TODO: What is the best way to specify other config that is part of schema?
        # e.g. invertedIndexConfig, shardingConfig, moduleConfig, vectorIndexConfig
        schema["properties"] = properties
        schema["class"] = self._db_config.index_name

        self._client.schema.create_class(schema)

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        host: str = 'http://weaviate:8080'
        index_name: str = 'Document'
        batch_config: Dict[str, Any] = field(
            default_factory=lambda: DEFAULT_BATCH_CONFIG
        )

    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {
                    'dataType': ['number[]'],
                },
                docarray.typing.ID: {'dataType': ['string']},
                bool: {'dataType': ['boolean']},
                int: {'dataType': ['int']},
                float: {'dataType': ['number']},
                str: {'dataType': ['text']},
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
        )

    def _del_items(self, doc_ids: Sequence[str]):
        return super()._del_items(doc_ids)

    def _filter(
        self, filter_query: Any, limit: int
    ) -> Union[DocumentArray, List[Dict]]:
        return super()._filter(filter_query, limit)

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocumentArray], List[List[Dict]]]:
        return super()._filter_batched(filter_queries, limit)

    def _find(self, query: np.ndarray, search_field: str, limit: int) -> FindResult:
        return super()._find(query, search_field, limit)

    def _find_batched(
        self, queries: Sequence[np.ndarray], search_field: str, limit: int
    ) -> _FindResultBatched:
        return super()._find_batched(queries, search_field, limit)

    def _get_items(self, doc_ids: Sequence[str]) -> List[Dict]:
        return super()._get_items(doc_ids)

    def _parse_document(self, document: Dict):
        doc = document.copy()

        # rewrite the id to __id
        document_id = doc.pop('id')
        doc['__id'] = document_id

        # TODO: find better way to get the vector column
        # find the vector column
        vector_column = None
        for k, v in doc.items():
            if isinstance(v, np.ndarray):
                vector_column = k
                break

        assert vector_column is not None, 'No vector column found'
        vector = doc.pop(vector_column)
        doc["embeddings"] = vector

        return doc

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        docs = self._transpose_col_value_dict(column_to_data)
        index_name = self._db_config.index_name

        with self._client.batch as batch:
            for doc in docs:
                parsed_doc = self._parse_document(doc)
                embeddings = parsed_doc.pop('embeddings')

                batch.add_data_object(
                    uuid=weaviate.util.generate_uuid5(parsed_doc, index_name),
                    data_object=parsed_doc,
                    class_name=index_name,
                    vector=embeddings,
                )

    def _text_search(self, query: str, search_field: str, limit: int) -> _FindResult:
        return super()._text_search(query, search_field, limit)

    def _text_search_batched(
        self, queries: Sequence[str], search_field: str, limit: int
    ) -> _FindResultBatched:
        return super()._text_search_batched(queries, search_field, limit)

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        return super().execute_query(query, *args, **kwargs)

    def num_docs(self) -> int:
        index_name = self._db_config.index_name
        result = self._client.query.aggregate(index_name).with_meta_count().do()

        total_docs = result["data"]["Aggregate"][index_name][0]["meta"]["count"]

        return total_docs

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        for allowed_type in WEAVIATE_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return np.ndarray

        if python_type in WEAVIATE_PY_TYPES:
            return python_type

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')
