from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Sequence, Tuple, Type, TypeVar, cast

import numpy as np
from voyager import BaseDoc, BaseDocIndex, DocList, VoyagerBaseDoc

from docarray.utils.find import _FindResult, _FindResultBatched

TSchema = TypeVar('TSchema', bound=VoyagerBaseDoc)
T = TypeVar('T', bound='VoyagerIndex')


@dataclass
class DBConfig(BaseDocIndex.DBConfig):
    default_column_config: Dict[Type, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RuntimeConfig(BaseDocIndex.RuntimeConfig):
    pass


class VoyagerIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)

        if not self._db_config or not self._db_config.existing_table:
            self._create_docs_table()

        self._setup_backend()

    def _create_docs_table(self):
        columns: List[Tuple[str, str]] = []
        for col, info in self._column_infos.items():
            if (
                col == 'id'
                or '__' in col
                or not info.db_type
                or info.db_type == np.ndarray
            ):
                continue
            columns.append((col, info.db_type))

        columns_str = ', '.join(f'{name} {type}' for name, type in columns)
        if columns_str:
            columns_str = ', ' + columns_str

        query = f'CREATE TABLE IF NOT EXISTS docs (doc_id INTEGER PRIMARY KEY, data BLOB{columns_str})'
        self._sqlite_cursor.execute(query)

    def _index(self, column_to_data: Dict[str, Any]):
        # Implement the indexing logic here
        # Example: Assume a simple case where you have a database table and you want to insert a new row
        self._insert_row_into_database(column_to_data)

    def _filter_by_parent_id(self, parent_id: str):
        # Implement the filter logic here
        # Example: Assume a simple case where you want to query rows in the database based on parent_id
        return self._query_rows_from_database_by_parent_id(parent_id)

    @property
    def index_name(self):
        return self._db_config.work_dir

    def _insert_row_into_database(self, column_to_data: Dict[str, Any]):
        # Placeholder logic: Insert a new row into the database
        # Adapt this according to your actual database backend
        print("Inserting row into the database:", column_to_data)

    def _query_rows_from_database_by_parent_id(self, parent_id: str):
        # Placeholder logic: Query rows from the database based on parent_id
        # Adapt this according to your actual database backend
        print("Querying rows from the database by parent_id:", parent_id)
        return []

    def add_documents(self, documents: DocList):
        vectors = [self.get_vector(doc) for doc in documents]
        self.add_items(vectors)
        self._num_docs += len(documents)

    def build(self):
        self.build_index()

    def build_query(self, query: Dict):
        return VoyagerQueryBuilder(self, query)

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        if isinstance(query, list):
            return self._execute_voyager_native_query(query)

        return self._execute_voyager_query_builder(query)

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> '_FindResultBatched':
        ids, distances = self._query_voyager(
            queries, k=limit, search_field=search_field
        )
        documents = [self.get_item(id_) for id_ in ids]
        distances_np = np.array(distances)

        return _FindResultBatched(documents, distances_np.tolist())

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> '_FindResult':
        query_batched = np.expand_dims(query, axis=0)
        docs, scores = self._find_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )
        return _FindResult(
            documents=docs[0], scores=NdArray._docarray_from_native(scores[0])
        )

    def _query_voyager(
        self,
        queries: np.ndarray,
        k: int,
        search_field: str = '',
    ) -> Tuple[List[str], List[float]]:
        result = self.query(queries, k=k, search_field=search_field)

        # Extracting ids and distances from the result
        ids = [doc['id'] for doc in result]
        distances = [doc['distance'] for doc in result]

        return ids, distances


class VoyagerQueryBuilder(BaseDocIndex.QueryBuilder):
    def __init__(self, document_index, query):
        super().__init__(document_index)
        self.query = query

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        ids, distances = self._query_voyager(
            queries, k=limit, search_field=search_field
        )

        documents = [self.get_item(id_) for id_ in ids]

        # Explicitly specify the type of distances to List[float]
        distances_list = distances.tolist()  # Assuming distances is a numpy array

        return _FindResultBatched(documents, distances_list)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        query_batched = np.expand_dims(query, axis=0)
        batched_result = self._find_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )

        # Assuming scores are available in batched_result
        scores = batched_result.scores

        return self._FindResult(documents=batched_result.documents, scores=scores)

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> DocList:
        result = self.execute_query(filter_query)

        ids = [doc['id'] for doc in result]
        embeddings = [doc['embedding'] for doc in result]

        docs = DocList.__class_getitem__(cast(Type[BaseDoc], self.out_schema))()
        for id_, embedding in zip(ids, embeddings):
            doc = self._doc_from_bytes(embedding)  # You need to implement this method
            doc.id = id_
            docs.append(doc)

        return docs

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> List[DocList]:
        # You can implement batched filtering logic here
        # For example, execute each filter query separately and combine the results
        raise NotImplementedError(
            f'{type(self)} does not support filter-only batched queries.'
            f' To perform post-filtering on a query, use'
            f' `build_query()` and `execute_query()`.'
        )

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        result = self.execute_query({'text_search': query, 'limit': limit})

        ids = [doc['id'] for doc in result]
        embeddings = [doc['embedding'] for doc in result]

        docs = DocList.__class_getitem__(cast(Type[BaseDoc], self.out_schema))()
        for id_, embedding in zip(ids, embeddings):
            doc = self._doc_from_bytes(embedding)  # You need to implement this method
            doc.id = id_
            docs.append(doc)

        return _FindResult(
            documents=docs,
            scores=[1.0] * len(docs),  # You may adjust the scores as needed
        )

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        # You can implement batched text search logic here
        # For example, execute each text search query separately and combine the results
        raise NotImplementedError(
            f'{type(self)} does not support text search batched queries.'
        )


class NdArray:
    @staticmethod
    def _docarray_from_native(data):
        """
        Convert a NumPy array to a document array.

        :param data: NumPy array
        :return: Document array
        """
        # Placeholder logic: Implement the actual conversion logic based on your requirements
        # For example, you can create a list of dictionaries where each dictionary represents a document
        # and contains key-value pairs corresponding to the document's fields and values.

        doc_array = []
        for row in data:
            # Assuming row is a NumPy array representing a document
            # Modify this based on the structure of your data
            doc = {
                'field1': row[0],
                'field2': row[1],
                # Add more fields as needed
            }
            doc_array.append(doc)

        return doc_array
