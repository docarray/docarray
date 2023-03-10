from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar, Union, cast

import numpy as np

from docarray import BaseDocument, DocumentArray
from docarray.doc_index.abstract_doc_index import BaseDocumentIndex, _FindResultBatched
from docarray.utils.find import FindResult, _FindResult

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='WeaviateDocumentIndex')

DEFAULT_SCHEMA = {
    "class": "Document",
    "properties": [
        {
            "name": "document_id",
            "dataType": ["string"],
            "description": "The unique identifier of the document.",
        },
        {
            "name": "text",
            "dataType": ["text"],
            "description": "The text of the document.",
        },
    ],
}

DEFAULT_BATCH_CONFIG = {
    "batch_size": 20,
    "dynamic": False,
    "timeout_retries": 3,
    "num_workers": 1,
}


class WeaviateDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs) -> None:
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(WeaviateDocumentIndex.DBConfig, self._db_config)

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        host: str = 'http://weaviate:8080'
        schema: Dict[str, Any] = field(default_factory=lambda: DEFAULT_SCHEMA)
        batch_config: Dict[str, Any] = field(
            default_factory=lambda: DEFAULT_BATCH_CONFIG
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

    def _index(self, docs: Sequence[TSchema]):
        return super()._index(docs)

    def _text_search(self, query: str, search_field: str, limit: int) -> _FindResult:
        return super()._text_search(query, search_field, limit)

    def _text_search_batched(
        self, queries: Sequence[str], search_field: str, limit: int
    ) -> _FindResultBatched:
        return super()._text_search_batched(queries, search_field, limit)

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        return super().execute_query(query, *args, **kwargs)

    def num_docs(self) -> int:
        return super().num_docs()

    def python_type_to_db_type(self, python_type: Type) -> Any:
        return super().python_type_to_db_type(python_type)
