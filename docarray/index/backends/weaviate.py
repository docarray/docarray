import base64
import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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
    cast,
)

import numpy as np
from pydantic import parse_obj_as
from typing_extensions import Literal

import docarray
from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.index.abstract import BaseDocIndex, FindResultBatched, _FindResultBatched
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.misc import import_library
from docarray.utils.find import FindResult, _FindResult

if TYPE_CHECKING:
    import weaviate
else:
    weaviate = import_library('weaviate')

TSchema = TypeVar('TSchema', bound=BaseDoc)
T = TypeVar('T', bound='WeaviateDocumentIndex')


DEFAULT_BATCH_CONFIG = {
    "batch_size": 20,
    "dynamic": False,
    "timeout_retries": 3,
    "num_workers": 1,
}

DEFAULT_BINARY_PATH = str(Path.home() / ".cache/weaviate-embedded/")
DEFAULT_PERSISTENCE_DATA_PATH = str(Path.home() / ".local/share/weaviate")


@dataclass
class EmbeddedOptions:
    persistence_data_path: str = os.environ.get(
        "XDG_DATA_HOME", DEFAULT_PERSISTENCE_DATA_PATH
    )
    binary_path: str = os.environ.get("XDG_CACHE_HOME", DEFAULT_BINARY_PATH)
    version: str = "latest"
    port: int = 6666
    hostname: str = "127.0.0.1"
    additional_env_vars: Optional[Dict[str, str]] = None


# TODO: add more types and figure out how to handle text vs string type
# see https://weaviate.io/developers/weaviate/configuration/datatypes
WEAVIATE_PY_VEC_TYPES = [list, np.ndarray, AbstractTensor]
WEAVIATE_PY_TYPES = [bool, int, float, str, docarray.typing.ID]

# "id" and "_id" are reserved names in weaviate so we need to use a different
# name for the id column in a BaseDocument
DOCUMENTID = "docarrayid"


class WeaviateDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs) -> None:
        """Initialize WeaviateDocumentIndex"""

        self.embedding_column: Optional[str] = None
        self.properties: Optional[List[str]] = None
        # keep track of the column name that contains the bytes
        # type because we will store them as a base64 encoded string
        # in weaviate
        self.bytes_columns: List[str] = []
        # keep track of the array columns that are not embeddings because we will
        # convert them to python lists before uploading to weaviate
        self.nonembedding_array_columns: List[str] = []
        super().__init__(db_config=db_config, **kwargs)
        self._db_config: WeaviateDocumentIndex.DBConfig = cast(
            WeaviateDocumentIndex.DBConfig, self._db_config
        )
        self._runtime_config: WeaviateDocumentIndex.RuntimeConfig = cast(
            WeaviateDocumentIndex.RuntimeConfig, self._runtime_config
        )

        if self._db_config.embedded_options:
            self._client = weaviate.Client(
                embedded_options=self._db_config.embedded_options
            )
        else:
            self._client = weaviate.Client(
                self._db_config.host, auth_client_secret=self._build_auth_credentials()
            )

        self._configure_client()
        self._validate_columns()
        self._set_embedding_column()
        self._set_properties()
        self._create_schema()

    @property
    def index_name(self):
        default_index_name = self._schema.__name__ if self._schema is not None else None
        if default_index_name is None:
            raise ValueError(
                'A WeaviateDocumentIndex must be typed with a Document type.'
                'To do so, use the syntax: WeaviateDocumentIndex[DocumentType]'
            )

        return self._db_config.index_name or default_index_name

    def _set_properties(self) -> None:
        field_overwrites = {"id": DOCUMENTID}

        self.properties = [
            field_overwrites.get(k, k)
            for k, v in self._column_infos.items()
            if v.config.get('is_embedding', False) is False
            and not safe_issubclass(v.docarray_type, AnyDocArray)
        ]

    def _validate_columns(self) -> None:
        # must have at most one column with property is_embedding=True
        # and that column must be of type WEAVIATE_PY_VEC_TYPES
        # TODO: update when https://github.com/weaviate/weaviate/issues/2424
        # is implemented and discuss best interface to signal which column(s)
        # should be used for embeddings
        num_embedding_columns = 0

        for column_name, column_info in self._column_infos.items():
            if column_info.config.get('is_embedding', False):
                num_embedding_columns += 1
                # if db_type is not 'number[]', then that means the type of the column in
                # the given schema is not one of WEAVIATE_PY_VEC_TYPES
                # note: the mapping between a column's type in the schema to a weaviate type
                # is handled by the python_type_to_db_type method
                if column_info.db_type != 'number[]':
                    raise ValueError(
                        f'Column {column_name} is marked as embedding but is not of type {WEAVIATE_PY_VEC_TYPES}'
                    )

        if num_embedding_columns > 1:
            raise ValueError(
                f'Only one column can be marked as embedding but found {num_embedding_columns} columns marked as embedding'
            )

    def _set_embedding_column(self) -> None:
        for column_name, column_info in self._column_infos.items():
            if column_info.config.get('is_embedding', False):
                self.embedding_column = column_name
                break

    def _configure_client(self) -> None:
        self._client.batch.configure(**self._runtime_config.batch_config)

    def _build_auth_credentials(self):
        dbconfig = self._db_config

        if dbconfig.auth_api_key:
            return weaviate.auth.AuthApiKey(api_key=dbconfig.auth_api_key)
        elif dbconfig.username and dbconfig.password:
            return weaviate.auth.AuthClientPassword(
                dbconfig.username, dbconfig.password, dbconfig.scopes
            )
        else:
            return None

    def configure(self, runtime_config=None, **kwargs) -> None:
        """
        Configure the WeaviateDocumentIndex.
        You can either pass a config object to `config` or pass individual config
        parameters as keyword arguments.
        If a configuration object is passed, it will replace the current configuration.
        If keyword arguments are passed, they will update the current configuration.

        :param runtime_config: the configuration to apply
        :param kwargs: individual configuration parameters
        """
        super().configure(runtime_config, **kwargs)
        self._configure_client()

    def _create_schema(self) -> None:
        schema: Dict[str, Any] = {}

        properties = []
        column_infos = self._column_infos

        for column_name, column_info in column_infos.items():
            # in weaviate, we do not create a property for the doc's embeddings
            if safe_issubclass(column_info.docarray_type, AnyDocArray):
                continue
            if column_name == self.embedding_column:
                continue
            if column_info.db_type == 'blob':
                self.bytes_columns.append(column_name)
            if column_info.db_type == 'number[]':
                self.nonembedding_array_columns.append(column_name)
            prop = {
                "name": column_name
                if column_name != 'id'
                else DOCUMENTID,  # in weaviate, id and _id is a reserved keyword
                "dataType": [column_info.db_type],
            }
            properties.append(prop)

        # TODO: What is the best way to specify other config that is part of schema?
        # e.g. invertedIndexConfig, shardingConfig, moduleConfig, vectorIndexConfig
        #       and configure replication
        # we will update base on user feedback
        schema["properties"] = properties
        schema["class"] = self.index_name

        if self._client.schema.exists(self.index_name):
            logging.warning(
                f"Found index {self.index_name} with schema {schema}. Will reuse existing schema."
            )
        else:
            self._client.schema.create_class(schema)

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of WeaviateDocumentIndex."""

        host: str = 'http://localhost:8080'
        index_name: Optional[str] = None
        username: Optional[str] = None
        password: Optional[str] = None
        scopes: List[str] = field(default_factory=lambda: ["offline_access"])
        auth_api_key: Optional[str] = None
        embedded_options: Optional[EmbeddedOptions] = None
        default_column_config: Dict[Any, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {},
                docarray.typing.ID: {},
                'string': {},
                'text': {},
                'int': {},
                'number': {},
                'boolean': {},
                'number[]': {},
                'blob': {},
            }
        )

        def __post_init__(self):
            # To prevent errors, it is important to capitalize the provided index name
            # when working with Weaviate, as it stores index names in a capitalized format.
            # Can't use .capitalize() because it modifies the whole string (See test).
            self.index_name = (
                self.index_name[0].upper() + self.index_name[1:]
                if self.index_name
                else None
            )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of WeaviateDocumentIndex."""

        batch_config: Dict[str, Any] = field(
            default_factory=lambda: DEFAULT_BATCH_CONFIG
        )

    def _del_items(self, doc_ids: Sequence[str]):
        has_matches = True

        operands = [
            {"path": [DOCUMENTID], "operator": "Equal", "valueString": doc_id}
            for doc_id in doc_ids
        ]
        where_filter = {
            "operator": "Or",
            "operands": operands,
        }

        # do a loop because there is a limit to how many objects can be deleted at
        # in a single query
        # see: https://weaviate.io/developers/weaviate/api/rest/batch#maximum-number-of-deletes-per-query
        while has_matches:
            results = self._client.batch.delete_objects(
                class_name=self.index_name,
                where=where_filter,
            )

            has_matches = results["results"]["matches"]

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        self._overwrite_id(filter_query)

        results = (
            self._client.query.get(self.index_name, self.properties)
            .with_additional("vector")
            .with_where(filter_query)
            .with_limit(limit)
            .do()
        )

        docs = results["data"]["Get"][self.index_name]

        return [self._parse_weaviate_result(doc) for doc in docs]

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        for filter_query in filter_queries:
            self._overwrite_id(filter_query)

        qs = [
            self._client.query.get(self.index_name, self.properties)
            .with_additional("vector")
            .with_where(filter_query)
            .with_limit(limit)
            .with_alias(f'query_{i}')
            for i, filter_query in enumerate(filter_queries)
        ]

        batched_results = self._client.query.multi_get(qs).do()

        return [
            [self._parse_weaviate_result(doc) for doc in batched_result]
            for batched_result in batched_results["data"]["Get"].values()
        ]

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ):
        """
        Find k-nearest neighbors of the query.

        :param query: query vector for KNN/ANN search. Has single axis.
        :param search_field: name of the field to search on
        :param limit: maximum number of documents to return per query
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug('Executing `find`')
        if search_field != '':
            raise ValueError(
                'Argument search_field is not supported for WeaviateDocumentIndex.\nSet search_field to an empty string to proceed.'
            )
        embedding_field = self._get_embedding_field()
        if isinstance(query, BaseDoc):
            query_vec = self._get_values_by_column([query], embedding_field)[0]
        else:
            query_vec = query
        query_vec_np = self._to_numpy(query_vec)
        docs, scores = self._find(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def _overwrite_id(self, where_filter):
        """
        Overwrite the id field in the where filter to DOCUMENTID
        if the "id" field is present in the path
        """
        for key, value in where_filter.items():
            if key == "path" and value == ["id"]:
                where_filter[key] = [DOCUMENTID]
            elif isinstance(value, dict):
                self._overwrite_id(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._overwrite_id(item)

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
        score_name: Literal["certainty", "distance"] = "certainty",
        score_threshold: Optional[float] = None,
    ) -> _FindResult:
        index_name = self.index_name
        if search_field:
            logging.warning(
                'The search_field argument is not supported for the WeaviateDocumentIndex and will be ignored.'
            )
        near_vector: Dict[str, Any] = {
            "vector": query,
        }
        if score_threshold:
            near_vector[score_name] = score_threshold

        results = (
            self._client.query.get(index_name, self.properties)
            .with_near_vector(
                near_vector,
            )
            .with_limit(limit)
            .with_additional([score_name, "vector"])
            .do()
        )

        docs, scores = self._format_response(
            results["data"]["Get"][index_name], score_name
        )
        return _FindResult(docs, parse_obj_as(NdArray, scores))

    def _format_response(
        self, results, score_name
    ) -> Tuple[List[Dict[Any, Any]], List[Any]]:
        """
        Format the response from Weaviate into a Tuple of DocList and scores
        """

        documents = []
        scores = []

        for result in results:
            score = result["_additional"][score_name]
            scores.append(score)

            document = self._parse_weaviate_result(result)
            documents.append(document)

        return documents, scores

    def find_batched(
        self,
        queries: Union[AnyTensor, DocList],
        search_field: str = '',
        limit: int = 10,
        **kwargs: Any,
    ) -> FindResultBatched:
        """Find documents in the index using nearest neighbor search.

        :param queries: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.) with a,
            or a DocList.
            If a tensor-like is passed, it should have shape (batch_size, vector_dim)
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return per query
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug('Executing `find_batched`')
        if search_field != '':
            raise ValueError(
                'Argument search_field is not supported for WeaviateDocumentIndex.\nSet search_field to an empty string to proceed.'
            )
        embedding_field = self._get_embedding_field()

        if isinstance(queries, Sequence):
            query_vec_list = self._get_values_by_column(queries, embedding_field)
            query_vec_np = np.stack(
                tuple(self._to_numpy(query_vec) for query_vec in query_vec_list)
            )
        else:
            query_vec_np = self._to_numpy(queries)

        da_list, scores = self._find_batched(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if len(da_list) > 0 and isinstance(da_list[0], List):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return FindResultBatched(documents=da_list, scores=scores)  # type: ignore

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
        score_name: Literal["certainty", "distance"] = "certainty",
        score_threshold: Optional[float] = None,
    ) -> _FindResultBatched:
        qs = []
        for i, query in enumerate(queries):
            near_vector: Dict[str, Any] = {"vector": query}

            if score_threshold:
                near_vector[score_name] = score_threshold

            q = (
                self._client.query.get(self.index_name, self.properties)
                .with_near_vector(near_vector)
                .with_limit(limit)
                .with_additional([score_name, "vector"])
                .with_alias(f'query_{i}')
            )

            qs.append(q)

        results = self._client.query.multi_get(qs).do()

        docs_and_scores = [
            self._format_response(result, score_name)
            for result in results["data"]["Get"].values()
        ]

        docs, scores = zip(*docs_and_scores)
        return _FindResultBatched(list(docs), list(scores))

    def _get_items(self, doc_ids: Sequence[str]) -> List[Dict]:
        # TODO: warn when doc_ids > QUERY_MAXIMUM_RESULTS after
        #       https://github.com/weaviate/weaviate/issues/2792
        #       is implemented
        operands = [
            {"path": [DOCUMENTID], "operator": "Equal", "valueString": doc_id}
            for doc_id in doc_ids
        ]
        where_filter = {
            "operator": "Or",
            "operands": operands,
        }

        results = (
            self._client.query.get(self.index_name, self.properties)
            .with_where(where_filter)
            .with_additional("vector")
            .do()
        )

        docs = [
            self._parse_weaviate_result(doc)
            for doc in results["data"]["Get"][self.index_name]
        ]

        return docs

    def _rewrite_documentid(self, document: Dict):
        doc = document.copy()

        # rewrite the id to DOCUMENTID
        document_id = doc.pop('id')
        doc[DOCUMENTID] = document_id

        return doc

    def _parse_weaviate_result(self, result: Dict) -> Dict:
        """
        Parse the result from weaviate to a format that is compatible with the schema
        that was used to initialize weaviate with.
        """

        result = result.copy()

        # rewrite the DOCUMENTID to id
        if DOCUMENTID in result:
            result['id'] = result.pop(DOCUMENTID)

        # take the vector from the _additional field
        if '_additional' in result and self.embedding_column:
            additional_fields = result.pop('_additional')
            if 'vector' in additional_fields:
                result[self.embedding_column] = additional_fields['vector']

        # convert any base64 encoded bytes column to bytes
        self._decode_base64_properties_to_bytes(result)

        return result

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        self._index_subindex(column_to_data)

        docs = self._transpose_col_value_dict(column_to_data)
        index_name = self.index_name

        with self._client.batch as batch:
            for doc in docs:
                parsed_doc = self._rewrite_documentid(doc)
                self._encode_bytes_columns_to_base64(parsed_doc)
                self._convert_nonembedding_array_to_list(parsed_doc)
                vector = (
                    parsed_doc.pop(self.embedding_column)
                    if self.embedding_column
                    else None
                )

                batch.add_data_object(
                    uuid=weaviate.util.generate_uuid5(parsed_doc, index_name),
                    data_object=parsed_doc,
                    class_name=index_name,
                    vector=vector,
                )

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        index_name = self.index_name
        bm25 = {"query": query, "properties": [search_field]}

        results = (
            self._client.query.get(index_name, self.properties)
            .with_bm25(**bm25)
            .with_limit(limit)
            .with_additional(["score", "vector"])
            .do()
        )

        docs, scores = self._format_response(
            results["data"]["Get"][index_name], "score"
        )

        return _FindResult(documents=docs, scores=parse_obj_as(NdArray, scores))

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        qs = []
        for i, query in enumerate(queries):
            bm25 = {"query": query, "properties": [search_field]}

            q = (
                self._client.query.get(self.index_name, self.properties)
                .with_bm25(**bm25)
                .with_limit(limit)
                .with_additional(["score", "vector"])
                .with_alias(f'query_{i}')
            )

            qs.append(q)

        results = self._client.query.multi_get(qs).do()

        docs_and_scores = [
            self._format_response(result, "score")
            for result in results["data"]["Get"].values()
        ]

        docs, scores = zip(*docs_and_scores)
        return _FindResultBatched(list(docs), list(scores))

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Execute a query on the WeaviateDocumentIndex.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        da_class = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))

        if isinstance(query, self.QueryBuilder):
            batched_results = self._client.query.multi_get(query._queries).do()
            batched_docs = batched_results["data"]["Get"].values()

            def f(doc):
                # TODO: use
                # return self._schema(**self._parse_weaviate_result(doc))
                # when https://github.com/weaviate/weaviate/issues/2858
                # is fixed
                return self._schema.from_view(self._parse_weaviate_result(doc))  # type: ignore

            results = [
                da_class([f(doc) for doc in batched_doc])
                for batched_doc in batched_docs
            ]
            return results if len(results) > 1 else results[0]

        # TODO: validate graphql query string before sending it to weaviate
        if isinstance(query, str):
            return self._client.query.raw(query)

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        index_name = self.index_name
        result = self._client.query.aggregate(index_name).with_meta_count().do()
        # TODO: decorator to check for errors
        total_docs = result["data"]["Aggregate"][index_name][0]["meta"]["count"]

        return total_docs

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        for allowed_type in WEAVIATE_PY_VEC_TYPES:
            if safe_issubclass(python_type, allowed_type):
                return 'number[]'

        py_weaviate_type_map = {
            docarray.typing.ID: 'string',
            str: 'text',
            int: 'int',
            float: 'number',
            bool: 'boolean',
            np.ndarray: 'number[]',
            bytes: 'blob',
        }

        for py_type, weaviate_type in py_weaviate_type_map.items():
            if safe_issubclass(python_type, py_type):
                return weaviate_type

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def build_query(self) -> BaseDocIndex.QueryBuilder:
        """
        Build a query for WeaviateDocumentIndex.
        :return: QueryBuilder object
        """
        return self.QueryBuilder(self)

    def _get_embedding_field(self):
        for colname, colinfo in self._column_infos.items():
            # no need to check for missing is_embedding attribute because this check
            # is done when the index is created
            if colinfo.config.get('is_embedding', None):
                return colname

        # just to pass mypy
        return ""

    def _encode_bytes_columns_to_base64(self, doc):
        for column in self.bytes_columns:
            if doc[column] is not None:
                doc[column] = base64.b64encode(doc[column]).decode("utf-8")

    def _decode_base64_properties_to_bytes(self, doc):
        for column in self.bytes_columns:
            if doc[column] is not None:
                doc[column] = base64.b64decode(doc[column])

    def _convert_nonembedding_array_to_list(self, doc):
        for column in self.nonembedding_array_columns:
            if doc[column] is not None:
                doc[column] = doc[column].tolist()

    def _filter_by_parent_id(self, id: str) -> Optional[List[str]]:
        results = (
            self._client.query.get(self._db_config.index_name, ['docarrayid'])
            .with_where(
                {'path': ['parent_id'], 'operator': 'Equal', 'valueString': f'{id}'}
            )
            .do()
        )

        ids = [
            res['docarrayid']
            for res in results['data']['Get'][self._db_config.index_name]
        ]
        return ids

    def _doc_exists(self, doc_id: str) -> bool:
        result = (
            self._client.query.get(self.index_name, ['docarrayid'])
            .with_where(
                {
                    "path": ['docarrayid'],
                    "operator": "Equal",
                    "valueString": f'{doc_id}',
                }
            )
            .do()
        )
        docs = result["data"]["Get"][self.index_name]
        return docs is not None and len(docs) > 0

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, document_index):
            self._queries = [
                document_index._client.query.get(
                    document_index.index_name, document_index.properties
                )
            ]

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            num_queries = len(self._queries)

            for i in range(num_queries):
                q = self._queries[i]
                if self._is_hybrid_query(q):
                    self._make_proper_hybrid_query(q)
                q.with_additional(["vector"]).with_alias(f'query_{i}')

            return self

        def _is_hybrid_query(self, query: weaviate.gql.get.GetBuilder) -> bool:
            """
            Checks if a query has been composed with both a with_bm25 and a with_near_vector verb
            """
            if not query._near_ask:
                return False
            else:
                return query._bm25 and query._near_ask._content.get("vector", None)

        def _make_proper_hybrid_query(
            self, query: weaviate.gql.get.GetBuilder
        ) -> weaviate.gql.get.GetBuilder:
            """
            Modifies a query to be a proper hybrid query.

            In weaviate, a query with with_bm25 and with_near_vector verb is not a hybrid query.
            We need to use the with_hybrid verb to make it a hybrid query.
            """

            text_query = query._bm25.query
            vector_query = query._near_ask._content["vector"]
            hybrid_query = weaviate.gql.get.Hybrid(
                query=text_query, vector=vector_query, alpha=0.5
            )

            query._bm25 = None
            query._near_ask = None
            query._hybrid = hybrid_query

        def _overwrite_id(self, where_filter):
            """
            Overwrite the id field in the where filter to DOCUMENTID
            if the "id" field is present in the path
            """
            for key, value in where_filter.items():
                if key == "path" and value == ["id"]:
                    where_filter[key] = [DOCUMENTID]
                elif isinstance(value, dict):
                    self._overwrite_id(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._overwrite_id(item)

        def find(
            self,
            query,
            score_name: Literal["certainty", "distance"] = "certainty",
            score_threshold: Optional[float] = None,
            **kwargs,
        ) -> Any:
            """
            Find k-nearest neighbors of the query.

            :param query: query vector for search. Has single axis.
            :param score_name: either `"certainty"` (default) or `"distance"`
            :param score_threshold: the threshold of the score
            :return: self
            """
            if kwargs.get('search_field'):
                logging.warning(
                    'The search_field argument is not supported for the WeaviateDocumentIndex and will be ignored.'
                )

            near_vector = {
                "vector": query,
            }
            if score_threshold:
                near_vector[score_name] = score_threshold

            self._queries[0] = self._queries[0].with_near_vector(near_vector)
            return self

        def find_batched(
            self,
            queries,
            score_name: Literal["certainty", "distance"] = "certainty",
            score_threshold: Optional[float] = None,
        ) -> Any:
            """Find k-nearest neighbors of the query vectors.

            :param queries: query vector for KNN/ANN search.
                Can be either a tensor-like (np.array, torch.Tensor, etc.) with a,
                or a DocList.
                If a tensor-like is passed, it should have shape `(batch_size, vector_dim)`
            :param score_name: either `"certainty"` (default) or `"distance"`
            :param score_threshold: the threshold of the score
            :return: self
            """
            adj_queries, adj_clauses = self._resize_queries_and_clauses(
                self._queries, queries
            )
            new_queries = []

            for query, clause in zip(adj_queries, adj_clauses):
                near_vector = {
                    "vector": clause,
                }
                if score_threshold:
                    near_vector[score_name] = score_threshold

                new_queries.append(query.with_near_vector(near_vector))

            self._queries = new_queries

            return self

        def filter(self, where_filter: Any) -> Any:
            """Find documents in the index based on a filter query
            :param where_filter: a filter
            :return: self
            """
            where_filter = where_filter.copy()
            self._overwrite_id(where_filter)
            self._queries[0] = self._queries[0].with_where(where_filter)
            return self

        def filter_batched(self, filters) -> Any:
            """Find documents in the index based on a filter query
            :param filters: filters
            :return: self
            """
            adj_queries, adj_clauses = self._resize_queries_and_clauses(
                self._queries, filters
            )
            new_queries = []

            for query, clause in zip(adj_queries, adj_clauses):
                clause = clause.copy()
                self._overwrite_id(clause)
                new_queries.append(query.with_where(clause))

            self._queries = new_queries

            return self

        def text_search(self, query: str, search_field: Optional[str] = None) -> Any:
            """Find documents in the index based on a text search query

            :param query: The text to search for
            :param search_field: name of the field to search on
            :return: self
            """
            bm25: Dict[str, Any] = {"query": query}
            if search_field:
                bm25["properties"] = [search_field]
            self._queries[0] = self._queries[0].with_bm25(**bm25)
            return self

        def text_search_batched(
            self, queries: Sequence[str], search_field: Optional[str] = None
        ) -> Any:
            """Find documents in the index based on a text search query

            :param queries: The texts to search for
            :param search_field: name of the field to search on
            :return: self
            """
            adj_queries, adj_clauses = self._resize_queries_and_clauses(
                self._queries, queries
            )
            new_queries = []

            for query, clause in zip(adj_queries, adj_clauses):
                bm25 = {"query": clause}
                if search_field:
                    bm25["properties"] = [search_field]
                new_queries.append(query.with_bm25(**bm25))

            self._queries = new_queries

            return self

        def limit(self, limit: int) -> Any:
            self._queries = [query.with_limit(limit) for query in self._queries]
            return self

        def _resize_queries_and_clauses(self, queries, clauses):
            """
            Adjust the length and content of queries and clauses so that we can compose
            them element-wise
            """
            num_clauses = len(clauses)
            num_queries = len(queries)

            # if there's only one clause, then we assume that it should be applied
            # to every query
            if num_clauses == 1:
                return queries, clauses * num_queries
            # if there's only one query, then we can lengthen it to match the number
            # of clauses
            elif num_queries == 1:
                return [copy.deepcopy(queries[0]) for _ in range(num_clauses)], clauses
            # if the number of queries and clauses is the same, then we can just
            # return them as-is
            elif num_clauses == num_queries:
                return queries, clauses
            else:
                raise ValueError(
                    f"Can't compose {num_clauses} clauses with {num_queries} queries"
                )
