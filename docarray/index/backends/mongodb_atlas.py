import collections
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    NamedTuple,
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

from docarray import BaseDoc, DocList, handler
from docarray.index.abstract import BaseDocIndex, _raise_not_composable
from docarray.index.backends.helper import _collect_query_required_args
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import _FindResult, _FindResultBatched

logger = logging.getLogger(__name__)
logger.addHandler(handler)


MAX_CANDIDATES = 10_000
OVERSAMPLING_FACTOR = 10
TSchema = TypeVar('TSchema', bound=BaseDoc)


class HybridResult(NamedTuple):
    """Adds breakdown of scores into vector and text components."""

    documents: Union[DocList, List[Dict[str, Any]]]
    scores: AnyTensor
    score_breakdown: Dict[str, List[Any]]


class MongoDBAtlasDocumentIndex(BaseDocIndex, Generic[TSchema]):
    """DocumentIndex backed by MongoDB Atlas Vector Store.

    MongoDB Atlas provides full Text, Vector, and Hybrid Search
    and can store structured data, text and vector indexes
    in the same Collection (Index).

    Atlas provides efficient index and search on vector embeddings
    using the Hierarchical Navigable Small Worlds (HNSW) algorithm.

    For documentation, see the following.
     * Text Search: https://www.mongodb.com/docs/atlas/atlas-search/atlas-search-overview/
     * Vector Search: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/
     * Hybrid Search: https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/
    """

    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        logger.info(f'{self.__class__.__name__} has been initialized')

    @property
    def index_name(self):
        """The name of the index/collection in the database.

        Note that in MongoDB Atlas, one has Collections (analogous to Tables),
        which can have Search Indexes. They are distinct.
        DocArray tends to consider them together.

        The index_name can be set when initializing MongoDBAtlasDocumentIndex.
        The easiest way is to pass index_name=<collection_name> as a kwarg.
        Otherwise, a rational default uses the name of the DocumentTypes that it contains.
        """

        if self._db_config.index_name is not None:
            return self._db_config.index_name
        else:
            # Create a reasonable default
            if not self._schema:
                raise ValueError(
                    'A MongoDBAtlasDocumentIndex must be typed with a Document type.'
                    'To do so, use the syntax: MongoDBAtlasDocumentIndex[DocumentType]'
                )
            schema_name = self._schema.__name__.lower()
            logger.debug(f"db_config.index_name was not set. Using {schema_name}")
            return schema_name

    @property
    def _database_name(self):
        return self._db_config.database_name

    @cached_property
    def _client(self):
        return self._connect_to_mongodb_atlas(
            atlas_connection_uri=self._db_config.mongo_connection_uri
        )

    @property
    def _collection(self):
        """MongoDB Collection"""
        return self._client[self._database_name][self.index_name]

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

        vector_fields: Optional[Dict[str, np.ndarray]]
        filters: Optional[List[Any]]
        text_searches: Optional[List[Any]]
        limit: int

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        """Compose complex queries containing vector search (find), text_search, and filters.

        Arguments to `find` are vectors of embeddings, text_search expects strings,
        and filters expect dicts of MongoDB Query Language (MDB).


        NOTE: When doing Hybrid Search, pay close attention to the interpretation and use of inputs,
        particularly when multiple calls are made of the same method (find, text_search, filter).
        * find (Vector Search):  Embedding vectors will be averaged. The penalty/weight defined in DBConfig will not change.
        * text_search: Individual searches are performed, each with the same penalty/weight.
        * filter:  Within Vector Search, performs efficient k-NN filtering with the Lucene engine
        """

        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, limit: int = 1, *args, **kwargs) -> Any:
            """Build a `Query` that can be passed to `execute_query`."""
            search_fields: Dict[str, np.ndarray] = collections.defaultdict(list)
            filters: List[Any] = []
            text_searches: List[Any] = []
            for method, kwargs in self._queries:
                if method == 'find':
                    search_field = kwargs['search_field']
                    search_fields[search_field].append(kwargs["query"])

                elif method == 'filter':
                    filters.append(kwargs)
                else:
                    text_searches.append(kwargs)

            vector_fields = {
                field: np.average(vectors, axis=0)
                for field, vectors in search_fields.items()
            }
            return MongoDBAtlasDocumentIndex.Query(
                vector_fields=vector_fields,
                filters=filters,
                text_searches=text_searches,
                limit=limit,
            )

        find = _collect_query_required_args('find', {'search_field', 'query'})
        filter = _collect_query_required_args('filter', {'query'})
        text_search = _collect_query_required_args(
            'text_search', {'search_field', 'query'}
        )

        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search_batched = _raise_not_composable('text_search_batched')

    def execute_query(
        self, query: Any, *args, score_breakdown=True, **kwargs
    ) -> Any:  # _FindResult:
        """Execute a Query on the database.

        :param query: the query to execute. The output of this Document index's `QueryBuilder.build()` method.
        :param args: positional arguments to pass to the query
        :param score_breakdown: Will provide breakdown of scores into text and vector components for Hybrid Searches.
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        if not isinstance(query, MongoDBAtlasDocumentIndex.Query):
            raise ValueError(
                "Expected MongoDBAtlasDocumentIndex.Query. Found {type(query)=}."
                "For native calls to MongoDBAtlasDocumentIndex, simply call filter()"
            )

        if len(query.vector_fields) > 1:
            self._logger.warning(
                f"{len(query.vector_fields)} embedding vectors have been provided to the query. They will be averaged."
            )
        if len(query.text_searches) > 1:
            self._logger.warning(
                f"{len(query.text_searches)} text searches will be performed, and each receive a ranked score."
            )

        # collect filters
        filters: List[Dict[str, Any]] = []
        for filter_ in query.filters:
            filters.append(filter_['query'])

        # check if hybrid search is needed.
        hybrid = len(query.vector_fields) + len(query.text_searches) > 1
        if hybrid:
            if len(query.vector_fields) > 1:
                raise NotImplementedError(
                    "Hybrid Search on multiple Vector Indexes has yet to be done."
                )
            pipeline = self._hybrid_search(
                query.vector_fields, query.text_searches, filters, query.limit
            )
        else:
            if query.text_searches:
                # it is a simple text search, perhaps with filters.
                text_stage = self._text_search_stage(**query.text_searches[0])
                pipeline = [
                    text_stage,
                    {"$match": {"$and": filters} if filters else {}},
                    {
                        '$project': self._project_fields(
                            extra_fields={"score": {'$meta': 'searchScore'}}
                        )
                    },
                    {"$limit": query.limit},
                ]
            elif query.vector_fields:
                # it is a simple vector search, perhaps with filters.
                assert (
                    len(query.vector_fields) == 1
                ), "Query contains more than one vector_field."
                field, vector_query = list(query.vector_fields.items())[0]
                pipeline = [
                    self._vector_search_stage(
                        query=vector_query,
                        search_field=field,
                        limit=query.limit,
                        filters=filters,
                    ),
                    {
                        '$project': self._project_fields(
                            extra_fields={"score": {'$meta': 'vectorSearchScore'}}
                        )
                    },
                ]
            # it is only a filter search.
            else:
                pipeline = [{"$match": {"$and": filters}}]

        with self._collection.aggregate(pipeline) as cursor:
            results, scores = self._mongo_to_docs(cursor)
        docs = self._dict_list_to_docarray(results)

        if hybrid and score_breakdown and results:
            score_breakdown = collections.defaultdict(list)
            score_fields = [key for key in results[0] if "score" in key]
            for res in results:
                score_breakdown["id"].append(res["id"])
                for sf in score_fields:
                    score_breakdown[sf].append(res[sf])
            logger.debug(score_breakdown)
            return HybridResult(
                documents=docs, scores=scores, score_breakdown=score_breakdown
            )

        return _FindResult(documents=docs, scores=scores)

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        mongo_connection_uri: str = 'localhost'
        index_name: Optional[str] = None
        database_name: Optional[str] = "default"
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: collections.defaultdict(
                dict,
                {
                    bson.BSONARR: {
                        'distance': 'COSINE',
                        'oversample_factor': OVERSAMPLING_FACTOR,
                        'max_candidates': MAX_CANDIDATES,
                        'indexed': False,
                        'index_name': None,
                        'penalty': 5,
                    },
                    bson.BSONSTR: {
                        'indexed': False,
                        'index_name': None,
                        'operator': 'phrase',
                        'penalty': 1,
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
        score = result.get("score", None)
        return result, score

    @staticmethod
    def _mongo_to_docs(mongo_docs: Generator[Dict, None, None]) -> List[dict]:
        docs = []
        scores = []
        for mongo_doc in mongo_docs:
            doc, score = MongoDBAtlasDocumentIndex._mongo_to_doc(mongo_doc)
            docs.append(doc)
            scores.append(score)

        return docs, scores

    def _get_oversampling_factor(self, search_field: str) -> int:
        return self._column_infos[search_field].config["oversample_factor"]

    def _get_max_candidates(self, search_field: str) -> int:
        return self._column_infos[search_field].config["max_candidates"]

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        """Add and Index Documents to the datastore

        The input format is aimed towards column vectors, which is not
        the natural fit for MongoDB Collections, but we have chosen
        not to override BaseDocIndex.index as it provides valuable validation.
        This may change in the future.

        :param column_to_data: is a dictionary from column name to a generator
        """
        self._index_subindex(column_to_data)
        docs: List[Dict[str, Any]] = []
        while True:
            try:
                doc = {key: next(column_to_data[key]) for key in column_to_data}
                mongo_doc = self._doc_to_mongo(doc)
                docs.append(mongo_doc)
            except StopIteration:
                break
        self._collection.insert_many(docs)

    def num_docs(self) -> int:
        """Return the number of indexed documents"""
        return self._collection.count_documents({})

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
        self._collection.delete_many(mg_filter)

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`. Duplicate `doc_ids` can be omitted in the output.
        """
        mg_filter = {"_id": {"$in": doc_ids}}
        docs = self._collection.find(mg_filter)
        docs, _ = self._mongo_to_docs(docs)

        if not docs:
            raise KeyError(f'No document with id {doc_ids} found')
        return docs

    def _reciprocal_rank_stage(self, search_field: str, score_field: str):
        penalty = self._column_infos[search_field].config["penalty"]
        projection_fields = {
            key: f"$docs.{key}" for key in self._column_infos.keys() if key != "id"
        }
        projection_fields["_id"] = "$docs._id"
        projection_fields[score_field] = 1

        return [
            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {
                "$addFields": {
                    score_field: {"$divide": [1.0, {"$add": ["$rank", penalty, 1]}]}
                }
            },
            {'$project': projection_fields},
        ]

    def _add_stage_to_pipeline(self, pipeline: List[Any], stage: Dict[str, Any]):
        if pipeline:
            pipeline.append(
                {"$unionWith": {"coll": self.index_name, "pipeline": stage}}
            )
        else:
            pipeline.extend(stage)
        return pipeline

    def _final_stage(self, scores_fields, limit):
        """Sum individual scores, sort, and apply limit."""
        doc_fields = self._column_infos.keys()
        grouped_fields = {
            key: {"$first": f"${key}"} for key in doc_fields if key != "_id"
        }
        best_score = {score: {'$max': f'${score}'} for score in scores_fields}
        final_pipeline = [
            {"$group": {"_id": "$_id", **grouped_fields, **best_score}},
            {
                "$project": {
                    **{doc_field: 1 for doc_field in doc_fields},
                    **{score: {"$ifNull": [f"${score}", 0]} for score in scores_fields},
                }
            },
            {
                "$addFields": {
                    "score": {"$add": [f"${score}" for score in scores_fields]},
                }
            },
            {"$sort": {"score": -1}},
            {"$limit": limit},
        ]
        return final_pipeline

    @staticmethod
    def _score_field(search_field: str, search_field_counts: Dict[str, int]):
        score_field = f"{search_field}_score"
        count = search_field_counts[search_field]
        if count > 1:
            score_field += str(count)
        return score_field

    def _hybrid_search(
        self,
        vector_queries: Dict[str, Any],
        text_queries: List[Dict[str, Any]],
        filters: Dict[str, Any],
        limit: int,
    ):
        hybrid_pipeline = []  # combined aggregate pipeline
        search_field_counts = collections.defaultdict(
            int
        )  # stores count of calls on same search field
        score_fields = []  # names given to scores of each search stage
        for search_field, query in vector_queries.items():
            search_field_counts[search_field] += 1
            vector_stage = self._vector_search_stage(
                query=query,
                search_field=search_field,
                limit=limit,
                filters=filters,
            )
            score_field = self._score_field(search_field, search_field_counts)
            score_fields.append(score_field)
            vector_pipeline = [
                vector_stage,
                *self._reciprocal_rank_stage(search_field, score_field),
            ]
            self._add_stage_to_pipeline(hybrid_pipeline, vector_pipeline)

        for kwargs in text_queries:
            search_field_counts[kwargs["search_field"]] += 1
            text_stage = self._text_search_stage(**kwargs)
            search_field = kwargs["search_field"]
            score_field = self._score_field(search_field, search_field_counts)
            score_fields.append(score_field)
            reciprocal_rank_stage = self._reciprocal_rank_stage(
                search_field, score_field
            )
            text_pipeline = [
                text_stage,
                {"$match": {"$and": filters} if filters else {}},
                {"$limit": limit},
                *reciprocal_rank_stage,
            ]
            self._add_stage_to_pipeline(hybrid_pipeline, text_pipeline)

        hybrid_pipeline += self._final_stage(score_fields, limit)
        return hybrid_pipeline

    def _vector_search_stage(
        self,
        query: np.ndarray,
        search_field: str,
        limit: int,
        filters: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        search_index_name = self._get_column_db_index(search_field)
        oversampling_factor = self._get_oversampling_factor(search_field)
        max_candidates = self._get_max_candidates(search_field)
        query = query.astype(np.float64).tolist()

        stage = {
            '$vectorSearch': {
                'index': search_index_name,
                'path': search_field,
                'queryVector': query,
                'numCandidates': min(limit * oversampling_factor, max_candidates),
                'limit': limit,
            }
        }
        if filters:
            stage['$vectorSearch']['filter'] = {"$and": filters}
        return stage

    def _text_search_stage(
        self,
        query: str,
        search_field: str,
    ) -> Dict[str, Any]:
        operator = self._column_infos[search_field].config["operator"]
        index = self._get_column_db_index(search_field)
        return {
            "$search": {
                "index": index,
                operator: {"query": query, "path": search_field},
            }
        }

    def _doc_exists(self, doc_id: str) -> bool:
        """
        Checks if a given document exists in the index.

        :param doc_id: The id of a document to check.
        :return: True if the document exists in the index, False otherwise.
        """
        doc = self._collection.find_one({"_id": doc_id})
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

        vector_search_stage = self._vector_search_stage(query, search_field, limit)

        pipeline = [
            vector_search_stage,
            {
                '$project': self._project_fields(
                    extra_fields={"score": {'$meta': 'vectorSearchScore'}}
                )
            },
        ]

        with self._collection.aggregate(pipeline) as cursor:
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

    def _get_column_db_index(self, column_name: str) -> Optional[str]:
        """
        Retrieve the index name associated with the specified column name.

        Parameters:
            column_name (str): The name of the column.

        Returns:
            Optional[str]: The index name associated with the specified column name, or None if not found.
        """
        index_name = self._column_infos[column_name].config.get("index_name")

        is_vector_index = safe_issubclass(
            self._column_infos[column_name].docarray_type, AbstractTensor
        )
        is_text_index = safe_issubclass(
            self._column_infos[column_name].docarray_type, str
        )

        if index_name is None or not isinstance(index_name, str):
            if is_vector_index:
                raise ValueError(
                    f'The column {column_name} for MongoDBAtlasDocumentIndex should be associated '
                    'with an Atlas Vector Index.'
                )
            elif is_text_index:
                raise ValueError(
                    f'The column {column_name} for MongoDBAtlasDocumentIndex should be associated '
                    'with an Atlas Index.'
                )
        if not (is_vector_index or is_text_index):
            raise ValueError(
                f'The column {column_name} for MongoDBAtlasDocumentIndex cannot be associated to an index'
            )

        return index_name

    def _project_fields(self, extra_fields: Dict[str, Any] = None) -> dict:
        """
        Create a projection dictionary to include all fields defined in the column information.

        Returns:
            dict: A dictionary where each field key from the column information is mapped to the value 1,
                indicating that the field should be included in the projection.
        """

        fields = {key: 1 for key in self._column_infos.keys() if key != "id"}
        fields["_id"] = 1
        if extra_fields:
            fields.update(extra_fields)
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
        with self._collection.find(filter_query, limit=limit) as cursor:
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
        text_stage = self._text_search_stage(query=query, search_field=search_field)

        pipeline = [
            text_stage,
            {
                '$project': self._project_fields(
                    extra_fields={'score': {'$meta': 'searchScore'}}
                )
            },
            {"$limit": limit},
        ]

        with self._collection.aggregate(pipeline) as cursor:
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
        with self._collection.find({"parent_id": id}, projection={"_id": 1}) as cursor:
            return [doc["_id"] for doc in cursor]
