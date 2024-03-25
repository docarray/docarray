import collections
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property

# from importlib.metadata import version
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import bson
import numpy as np
from pymongo import MongoClient

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex, _raise_not_composable
from docarray.index.backends.helper import _collect_query_required_args
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import FindResult, _FindResult, _FindResultBatched

# from pymongo.driver_info import DriverInfo


MAX_CANDIDATES = 10_000
OVERSAMPLING_FACTOR = 10
TSchema = TypeVar('TSchema', bound=BaseDoc)


class MongoAtlasDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._create_indexes()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @property
    def _collection(self):
        if self._is_subindex:
            return self._ori_schema.__name__
        return self._schema.__name__

    @property
    def _database_name(self):
        return self._db_config.database_name

    @cached_property
    def _client(self):
        return self._connect_to_mongodb_atlas(
            atlas_connection_uri=self._db_config.mongo_connection_uri
        )

    @property
    def _doc_collection(self):
        return self._client[self._database_name][self._collection]

    @staticmethod
    def _connect_to_mongodb_atlas(atlas_connection_uri: str):
        """
        Establish a connection to MongoDB Atlas.
        """

        client = MongoClient(
            atlas_connection_uri,
            # driver=DriverInfo(name="docarray", version=version("docarray"))
        )
        return client

    def _create_indexes(self):
        """Create a new index in the MongoDB database if it doesn't already exist."""

    def _check_index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists in the MongoDB Atlas database.

        :param index_name: The name of the index.
        :return: True if the index exists, False otherwise.
        """

    @dataclass
    class Query:
        """Dataclass describing a query."""

        vector_field: Optional[str]
        vector_query: Optional[np.ndarray]
        filters: Optional[List[Any]]  # TODO: define a type
        text_searches: Optional[List[Any]]  # TODO: define a type
        limit: int

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, limit: int) -> Any:
            """Build the query object."""
            search_field = None
            vectors = []
            filters = []
            text_searches = []
            for method, kwargs in self._queries:
                if method == 'find':
                    if search_field and kwargs['search_field'] != search_field:
                        raise ValueError(
                            f'Trying to call .find for search_field = {kwargs["search_field"]}, but '
                            f'previously {search_field} was used. Only a single '
                            f'field might be used in chained calls.'
                        )

                    search_field = kwargs['search_field']
                    vectors.append(kwargs["query"])

                elif method == 'filter':
                    filters.append(kwargs)
                else:
                    text_searches.append(kwargs)

            vector = np.average(vectors, axis=0) if vectors else None
            return MongoAtlasDocumentIndex.Query(
                vector_field=search_field,
                vector_query=vector,
                filters=filters,
                text_searches=text_searches,
                limit=limit,
            )

        find = _collect_query_required_args('find', {'search_field', 'query'})
        filter = _collect_query_required_args('filter', {'query'})
        # it is included in filter method.
        text_search = _raise_not_composable('text_search')
        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search_batched = _raise_not_composable('text_search_batched')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        mongo_connection_uri: str = 'localhost'
        index_name: Optional[str] = None
        database_name: Optional[str] = "default"
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(
                dict,
                {
                    bson.BSONARR: {
                        'algorithm': 'KNN',
                        'distance': 'COSINE',
                        'oversample_factor': OVERSAMPLING_FACTOR,
                        'max_candidates': MAX_CANDIDATES,
                    },
                },
            )
        )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        ...

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """

        type_map = {
            int: bson.BSONNUM,
            float: bson.BSONDEC,
            collections.OrderedDict: bson.BSONOBJ,
            str: bson.BSONSTR,
            bytes: bson.BSONBIN,
            dict: bson.BSONOBJ,
            np.ndarray: bson.BSONARR,
            AbstractTensor: bson.BSONARR,
        }

        for py_type, mongo_types in type_map.items():
            if safe_issubclass(python_type, py_type):
                return mongo_types
        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _doc_to_mongo(self, doc):
        result = doc.copy()

        for name in result:
            if self._column_infos[name].db_type == bson.BSONARR:
                result[name] = list(result[name])

        result["_id"] = result.pop("id")
        return result

    def _docs_to_mongo(self, docs):
        return [self._doc_to_mongo(doc) for doc in docs]

    @staticmethod
    def _mongo_to_doc(mongo_doc: dict) -> dict:
        result = mongo_doc.copy()
        result["id"] = result.pop("_id")
        score = result.pop("score", None)
        return result, score

    @staticmethod
    def _mongo_to_docs(mongo_docs: Generator[Dict, None, None]) -> List[dict]:
        docs = []
        scores = []
        for mongo_doc in mongo_docs:
            doc, score = MongoAtlasDocumentIndex._mongo_to_doc(mongo_doc)
            docs.append(doc)
            scores.append(score)

        return docs, scores

    def _get_oversampling_factor(self, search_field: str) -> int:
        return self._column_infos[search_field].config["oversample_factor"]

    def _get_max_candidates(self, search_field: str) -> int:
        return self._column_infos[search_field].config["max_candidates"]

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        """index a document into the store"""
        # `column_to_data` is a dictionary from column name to a generator
        # that yields the data for that column.
        # If you want to work directly on documents, you can implement index() instead
        # If you implement index(), _index() only needs a dummy implementation.
        self._index_subindex(column_to_data)
        docs: List[Dict[str, Any]] = []
        while True:
            try:
                doc = {key: next(column_to_data[key]) for key in column_to_data}
                mongo_doc = self._doc_to_mongo(doc)
                docs.append(mongo_doc)
            except StopIteration:
                break
        self._doc_collection.insert_many(docs)

    def num_docs(self) -> int:
        """Return the number of indexed documents"""
        return self._doc_collection.count_documents({})

    @property
    def _is_index_empty(self) -> bool:
        """
        Check if index is empty by comparing the number of documents to zero.
        :return: True if the index is empty, False otherwise.
        """
        return self.num_docs() == 0

    def _del_items(self, doc_ids: Sequence[str]) -> None:
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        mg_filter = {"_id": {"$in": doc_ids}}
        self._doc_collection.delete_many(mg_filter)

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`. Duplicate `doc_ids` can be omitted in the output.
        """
        mg_filter = {"_id": {"$in": doc_ids}}
        docs = self._doc_collection.find(mg_filter)
        docs, _ = self._mongo_to_docs(docs)

        if not docs:
            raise KeyError(f'No document with id {doc_ids} found')
        return docs

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Execute a query on the database.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """

        filters: List[Dict[str, Any]] = []

        for filter_ in query.filters:
            filters.append(self._compute_filter_query(**filter_))

        for filter_ in query.text_searches:
            filters.append(self._compute_text_search_query(**filter_))

        if query.vector_field:
            pipeline = [
                self._compute_vector_search(
                    query=query.vector_query,
                    search_field=query.vector_field,
                    limit=query.limit,
                    filters=filters,
                ),
                {'$project': self._project_fields()},
            ]
        else:
            pipeline = [{"$match": {"$and": filters}}, {"$limit": query.limit}]

        with self._doc_collection.aggregate(pipeline) as cursor:
            docs, scores = self._mongo_to_docs(cursor)

        docs = self._dict_list_to_docarray(docs)
        return FindResult(documents=docs, scores=scores)

    def _compute_vector_search(
        self,
        query: np.ndarray,
        search_field: str,
        limit: int,
        filters: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:

        index_name = self._get_column_index(search_field)
        oversampling_factor = self._get_oversampling_factor(search_field)
        max_candidates = self._get_max_candidates(search_field)
        query = query.astype(np.float64).tolist()

        return {
            '$vectorSearch': {
                'index': index_name,
                'path': search_field,
                'queryVector': query,
                'numCandidates': min(limit * oversampling_factor, max_candidates),
                'limit': limit,
                'filter': {"$and": filters} if filters else None,
            }
        }

    def _compute_filter_query(
        self,
        query: Any,
    ) -> Dict[str, Any]:
        return query

    def _compute_text_search_query(
        self,
        query: str,
        search_field: str = '',
    ) -> Dict[str, Any]:
        return {
            search_field: {
                '$in': query,
            }
        }

    def _doc_exists(self, doc_id: str) -> bool:
        """
        Checks if a given document exists in the index.

        :param doc_id: The id of a document to check.
        :return: True if the document exists in the index, False otherwise.
        """
        doc = self._doc_collection.find_one({"_id": doc_id})
        return bool(doc)

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        """Find documents in the index

        :param query: query vector for KNN/ANN search. Has single axis.
        :param limit: maximum number of documents to return per query
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        query = query.astype(np.float64).tolist()
        index_name = self._get_column_index(search_field)

        oversampling_factor = self._get_oversampling_factor(search_field)
        max_candidates = self._get_max_candidates(search_field)

        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'path': search_field,
                    'queryVector': query,
                    'numCandidates': min(limit * oversampling_factor, max_candidates),
                    'limit': limit,
                }
            },
            {'$project': self._project_fields()},
        ]

        with self._doc_collection.aggregate(pipeline) as cursor:
            documents, scores = self._mongo_to_docs(cursor)

        return _FindResult(documents=documents, scores=scores)

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        """Find documents in the index

        :param queries: query vectors for KNN/ANN search.
            Has shape (batch_size, vector_dim)
        :param limit: maximum number of documents to return
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        docs, scores = [], []
        for query in queries:
            results = self._find(query=query, search_field=search_field, limit=limit)
            docs.append(results.documents)
            scores.append(results.scores)

        return _FindResultBatched(documents=docs, scores=scores)

    def _get_column_index(self, column_name: str) -> Optional[str]:
        """
        Retrieve the index name associated with the specified column name.

        Parameters:
            column_name (str): The name of the column.

        Returns:
            Optional[str]: The index name associated with the specified column name, or None if not found.
        """
        return self._column_infos[column_name].config.get("index_name")

    def _project_fields(self) -> dict:
        """
        Create a projection dictionary to include all fields defined in the column information.

        Returns:
            dict: A dictionary where each field key from the column information is mapped to the value 1,
                indicating that the field should be included in the projection.
        """
        fields = {key: 1 for key in self._column_infos.keys() if key != "_id"}
        fields["score"] = {'$meta': 'vectorSearchScore'}
        return fields

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> Union[DocList, List[Dict]]:
        """Find documents in the index based on a filter query

        :param filter_query: the DB specific filter query to execute
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        with self._doc_collection.find(filter_query, limit=limit) as cursor:
            return self._mongo_to_docs(cursor)[0]

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> Union[List[DocList], List[List[Dict]]]:
        """Find documents in the index based on multiple filter queries.
        Each query is considered individually, and results are returned per query.

        :param filter_queries: the DB specific filter queries to execute
        :param limit: maximum number of documents to return per query
        :return: List of DocLists containing the documents that match the filter
            queries
        """
        return [self._filter(query, limit) for query in filter_queries]

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        """Find documents in the index based on a text search query

        :param query: The text to search for
        :param limit: maximum number of documents to return
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        with self._doc_collection.find(
            {search_field: {'$regex': query}}, limit=limit
        ) as cursor:
            documents, scores = self._mongo_to_docs(cursor)

        return _FindResult(documents=documents, scores=scores)

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        """Find documents in the index based on a text search query

        :param queries: The texts to search for
        :param limit: maximum number of documents to return per query
        :param search_field: name of the field to search on
        :return: a named tuple containing `documents` and `scores`
        """
        # NOTE: in standard implementations,
        # `search_field` is equal to the column name to search on
        documents, scores = [], []
        for query in queries:
            results = self._text_search(
                query=query, search_field=search_field, limit=limit
            )
            documents.append(results.documents)
            scores.append(results.scores)
        return _FindResultBatched(documents=documents, scores=scores)

    def _filter_by_parent_id(self, id: str) -> Optional[List[str]]:
        """Filter the ids of the subindex documents given id of root document.

        :param id: the root document id to filter by
        :return: a list of ids of the subindex documents
        """
        with self._doc_collection.find(
            {"parent_id": id}, projection={"_id": 1}
        ) as cursor:
            return [doc["_id"] for doc in cursor]
