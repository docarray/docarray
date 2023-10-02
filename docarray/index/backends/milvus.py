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
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.index.abstract import (
    BaseDocIndex,
    _raise_not_composable,
    _raise_not_supported,
)
from docarray.index.backends.helper import _collect_query_args
from docarray.typing import AnyTensor, NdArray
from docarray.typing.id import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import (
    FindResult,
    FindResultBatched,
    _FindResult,
    _FindResultBatched,
)

if TYPE_CHECKING:
    from pymilvus import (  # type: ignore[import]
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        Hits,
        connections,
        utility,
    )
else:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        Hits,
        connections,
        utility,
    )

MAX_LEN = 65_535  # Maximum length that Milvus allows for a VARCHAR field
VALID_METRICS = ['L2', 'IP']
VALID_INDEX_TYPES = [
    'FLAT',
    'IVF_FLAT',
    'IVF_SQ8',
    'IVF_PQ',
    'HNSW',
    'ANNOY',
    'DISKANN',
]

TSchema = TypeVar('TSchema', bound=BaseDoc)


class MilvusDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        """Initialize MilvusDocumentIndex"""
        super().__init__(db_config=db_config, **kwargs)
        self._db_config: MilvusDocumentIndex.DBConfig = cast(
            MilvusDocumentIndex.DBConfig, self._db_config
        )
        self._runtime_config: MilvusDocumentIndex.RuntimeConfig = cast(
            MilvusDocumentIndex.RuntimeConfig, self._runtime_config
        )

        self._client = connections.connect(
            db_name="default",
            host=self._db_config.host,
            port=self._db_config.port,
            user=self._db_config.user,
            password=self._db_config.password,
            token=self._db_config.token,
        )

        self._validate_columns()
        self._field_name = self._get_vector_field_name()
        self._collection = self._create_or_load_collection()
        self._build_index()
        self._collection.load()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of MilvusDocumentIndex.

        :param index_name: The name of the index in the Milvus database. If not provided, default index name will be used.
        :param collection_description: Description of the collection in the database.
        :param host: Hostname of the server where the database resides. Default is 'localhost'.
        :param port: Port number used to connect to the database. Default is 19530.
        :param user: User for the database. Can be an empty string if no user is required.
        :param password: Password for the specified user. Can be an empty string if no password is required.
        :param token: Token for secure connection. Can be an empty string if no token is required.
        :param consistency_level: The level of consistency for the database session. Default is 'Session'.
        :param search_params: Dictionary containing parameters for search operations,
            default has a single key 'params' with 'nprobe' set to 10.
        :param serialize_config: Dictionary containing configuration for serialization,
            default is {'protocol': 'protobuf'}.
        :param default_column_config: Dictionary that defines the default configuration
            for each data type column.
        """

        index_name: Optional[str] = None
        collection_description: str = ""
        host: str = "localhost"
        port: int = 19530
        user: Optional[str] = ""
        password: Optional[str] = ""
        token: Optional[str] = ""
        consistency_level: str = 'Session'
        search_params: Dict = field(
            default_factory=lambda: {
                "params": {"nprobe": 10},
            }
        )
        serialize_config: Dict = field(default_factory=lambda: {"protocol": "protobuf"})
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(
                dict,
                {
                    DataType.FLOAT_VECTOR: {
                        'index_type': 'IVF_FLAT',
                        'metric_type': 'L2',
                        'params': {"nlist": 1024},
                    },
                },
            )
        )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of RedisDocumentIndex.

        :param batch_size: Batch size for index/get/del.
        """

        batch_size: int = 100

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            return self._queries

        find = _collect_query_args('find')
        filter = _collect_query_args('filter')
        text_search = _raise_not_supported('text_search')
        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search_batched = _raise_not_supported('text_search_batched')

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type, or None if ``python_type``
        is not supported.
        """
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
            return DataType.VARCHAR

        for py_type, db_type in type_map.items():
            if safe_issubclass(python_type, py_type):
                return db_type

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _create_or_load_collection(self) -> Collection:
        """
        This function initializes or retrieves a Milvus collection with a specified schema,
        storing documents as serialized data and using the document's ID as the collection's ID
        , while inheriting other schema properties from the indexer's schema.

        !!! note
            Milvus framework currently only supports a single vector column, and only one vector
            column can store in the schema (others are stored in the serialized data)
        """

        if not utility.has_collection(self.index_name):
            fields = [
                FieldSchema(
                    name="serialized",
                    dtype=DataType.VARCHAR,
                    max_length=MAX_LEN,
                ),
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=MAX_LEN,
                ),
            ]
            for column_name, info in self._column_infos.items():
                if (
                    column_name != 'id'
                    and not (
                        info.db_type == DataType.FLOAT_VECTOR
                        and column_name
                        != self._field_name  # Only store one vector field as a column
                    )
                    and not safe_issubclass(info.docarray_type, AnyDocArray)
                ):
                    field_dict: Dict[str, Any] = {}
                    if info.db_type == DataType.VARCHAR:
                        field_dict = {'max_length': MAX_LEN}
                    elif info.db_type == DataType.FLOAT_VECTOR:
                        field_dict = {'dim': info.n_dim or info.config.get('dim')}

                    fields.append(
                        FieldSchema(
                            name=column_name,
                            dtype=info.db_type,
                            is_primary=False,
                            **field_dict,
                        )
                    )

            self._logger.info("Collection has been created")
            return Collection(
                name=self.index_name,
                schema=CollectionSchema(
                    fields=fields,
                    description=self._db_config.collection_description,
                ),
                using='default',
            )

        return Collection(self.index_name)

    def _validate_columns(self):
        """
        Validates whether the data schema includes at least one vector column used
        for embedding (as required by Milvus), and ensures that dimension information
        is specified for that column.
        """
        vector_columns = sum(
            safe_issubclass(info.docarray_type, AbstractTensor)
            and info.config.get('is_embedding', False)
            for info in self._column_infos.values()
        )
        if vector_columns == 0:
            raise ValueError(
                "Unable to find any vector columns. Please make sure that at least one "
                "column is of a vector type with the is_embedding=True attribute specified."
            )
        elif vector_columns > 1:
            raise ValueError("Specifying multiple vector fields is not supported.")

        for column, info in self._column_infos.items():
            if info.config.get('is_embedding') and (
                not info.n_dim and not info.config.get('dim')
            ):
                raise ValueError(
                    f"The dimension information is missing for the column '{column}', which is of vector type."
                )

    @property
    def index_name(self):
        default_index_name = (
            self._schema.__name__.lower() if self._schema is not None else None
        )
        if default_index_name is None:
            err_msg = (
                'A MilvusDocumentIndex must be typed with a Document type. '
                'To do so, use the syntax: MilvusDocumentIndex[DocumentType]'
            )

            self._logger.error(err_msg)
            raise ValueError(err_msg)
        index_name = self._db_config.index_name or default_index_name
        self._logger.debug(f'Retrieved index name: {index_name}')
        return index_name

    @property
    def out_schema(self) -> Type[BaseDoc]:
        """Return the real schema of the index."""
        if self._is_subindex:
            return self._ori_schema
        return cast(Type[BaseDoc], self._schema)

    def _build_index(self):
        """
        Sets up an index configuration for a specific column index, which is
        required by the Milvus backend.
        """

        existing_indices = [index.field_name for index in self._collection.indexes]
        if self._field_name in existing_indices:
            return

        index_type = self._column_infos[self._field_name].config['index_type'].upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"Invalid index type '{index_type}' provided. "
                f"Must be one of: {', '.join(VALID_INDEX_TYPES)}"
            )
        metric_type = (
            self._column_infos[self._field_name].config.get('space', '').upper()
        )
        if metric_type not in VALID_METRICS:
            self._logger.warning(
                f"Invalid or no distance metric '{metric_type}' was provided. "
                f"Should be one of: {', '.join(VALID_INDEX_TYPES)}. "
                f"Default distance metric will be used."
            )
            metric_type = self._column_infos[self._field_name].config['metric_type']

        index = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": self._column_infos[self._field_name].config['params'],
        }

        self._collection.create_index(self._field_name, index)
        self._logger.info(
            f"Index for the field '{self._field_name}' has been successfully created"
        )

    def _get_vector_field_name(self):
        for column, info in self._column_infos.items():
            if info.db_type == DataType.FLOAT_VECTOR and info.config.get(
                'is_embedding'
            ):
                return column
        return ''

    @staticmethod
    def _get_batches(docs, batch_size):
        """Yield successive batch_size batches from docs."""
        for i in range(0, len(docs), batch_size):
            yield docs[i : i + batch_size]

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

        n_docs = 1 if isinstance(docs, BaseDoc) else len(docs)
        self._logger.debug(f'Indexing {n_docs} documents')
        docs = self._validate_docs(docs)
        self._update_subindex_data(docs)
        data_by_columns = self._get_col_value_dict(docs)
        self._index_subindex(data_by_columns)

        positions: Dict[str, int] = {
            info.name: num for num, info in enumerate(self._collection.schema.fields)
        }

        for batch in self._get_batches(
            docs, batch_size=self._runtime_config.batch_size
        ):
            entities: List[List[Any]] = [
                [] for _ in range(len(self._collection.schema))
            ]
            for doc in batch:
                # "serialized" will always be in the first position
                entities[0].append(doc.to_base64(**self._db_config.serialize_config))
                for schema_field in self._collection.schema.fields:
                    if schema_field.name == 'serialized':
                        continue
                    column_value = self._get_values_by_column([doc], schema_field.name)[
                        0
                    ]
                    if schema_field.dtype == DataType.FLOAT_VECTOR:
                        column_value = self._map_embedding(column_value)

                    entities[positions[schema_field.name]].append(column_value)
            self._collection.insert(entities)

        self._collection.flush()
        self._logger.info(f"{len(docs)} documents has been indexed")

    def _filter_by_parent_id(self, id: str) -> Optional[List[str]]:
        """Filter the ids of the subindex documents given id of root document.

        :param id: the root document id to filter by
        :return: a list of ids of the subindex documents
        """
        docs = self._filter(filter_query=f"parent_id == '{id}'", limit=self.num_docs())
        return [doc.id for doc in docs]  # type: ignore[union-attr]

    def num_docs(self) -> int:
        """
        Get the number of documents.

        !!! note
             Cannot use Milvus' num_entities method because it's not precise
             especially after delete ops (#15201 issue in Milvus)
        """

        self._collection.load()

        result = self._collection.query(
            expr=self._always_true_expr("id"),
            offset=0,
            output_fields=["serialized"],
        )

        return len(result)

    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :param raw: if raw, output the new_schema type (with parent id)
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`.
                Duplicate `doc_ids` can be omitted in the output.
        """

        self._collection.load()
        results: List[Dict] = []
        for batch in self._get_batches(
            doc_ids, batch_size=self._runtime_config.batch_size
        ):
            results.extend(
                self._collection.query(
                    expr="id in " + str([id for id in batch]),
                    offset=0,
                    output_fields=["serialized"],
                    consistency_level=self._db_config.consistency_level,
                )
            )

        self._collection.release()

        return self._docs_from_query_response(results)

    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        self._collection.load()
        for batch in self._get_batches(
            doc_ids, batch_size=self._runtime_config.batch_size
        ):
            self._collection.delete(
                expr="id in " + str([id for id in batch]),
                consistency_level=self._db_config.consistency_level,
            )
        self._logger.info(f"{len(doc_ids)} documents has been deleted")

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> Union[DocList, List[Dict]]:
        """
        Filters the index based on the given filter query.

        :param filter_query: The filter condition.
        :param limit: The maximum number of results to return.
        :return: Filter results.
        """

        self._collection.load()

        result = self._collection.query(
            expr=filter_query,
            offset=0,
            limit=min(limit, self.num_docs()),
            output_fields=["serialized"],
        )

        self._collection.release()

        return self._docs_from_query_response(result)

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> Union[List[DocList], List[List[Dict]]]:
        """
        Filters the index based on the given batch of filter queries.

        :param filter_queries: The filter conditions.
        :param limit: The maximum number of results to return for each filter query.
        :return: Filter results.
        """
        return [
            self._filter(filter_query=query, limit=limit) for query in filter_queries
        ]

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        """index a document into the store"""
        raise NotImplementedError()

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find documents in the index using nearest neighbor search.

        :param query: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.)
            with a single axis, or a Document
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `find` for search field {search_field}')
        if search_field != '':
            raise ValueError(
                'Argument search_field is not supported for MilvusDocumentIndex.'
                'Set search_field to an empty string to proceed.'
            )

        search_field = self._field_name
        if isinstance(query, BaseDoc):
            query_vec = self._get_values_by_column([query], search_field)[0]
        else:
            query_vec = query
        query_vec_np = self._to_numpy(query_vec)
        docs, scores = self._find(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )

        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        """
        Conducts a search on the index.

        :param query: The vector query to search.
        :param limit: The maximum number of results to return.
        :param search_field: The field to search the query.
        :return: Search results.
        """

        return self._hybrid_search(query=query, limit=limit, search_field=search_field)

    def _hybrid_search(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
        expr: Optional[str] = None,
    ):
        """
        Conducts a hybrid search on the index.

        :param query: The vector query to search.
        :param limit: The maximum number of results to return.
        :param search_field: The field to search the query.
        :param expr: Boolean expression used for filtering.
        :return: Search results.
        """
        self._collection.load()

        results = self._collection.search(
            data=[query],
            anns_field=search_field,
            param=self._db_config.search_params,
            limit=limit,
            offset=0,
            expr=expr,
            output_fields=["serialized"],
            consistency_level=self._db_config.consistency_level,
        )

        self._collection.release()

        results = next(iter(results), None)  # Only consider the first element

        return self._docs_from_find_response(results)

    def find_batched(
        self,
        queries: Union[AnyTensor, DocList],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
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
        self._logger.debug(f'Executing `find_batched` for search field {search_field}')

        if search_field:
            if '__' in search_field:
                fields = search_field.split('__')
                if issubclass(self._schema._get_field_annotation(fields[0]), AnyDocArray):  # type: ignore
                    return self._subindices[fields[0]].find_batched(
                        queries,
                        search_field='__'.join(fields[1:]),
                        limit=limit,
                        **kwargs,
                    )
        if search_field != '':
            raise ValueError(
                'Argument search_field is not supported for MilvusDocumentIndex.'
                'Set search_field to an empty string to proceed.'
            )
        search_field = self._field_name
        if isinstance(queries, Sequence):
            query_vec_list = self._get_values_by_column(queries, search_field)
            query_vec_np = np.stack(
                tuple(self._to_numpy(query_vec) for query_vec in query_vec_list)
            )
        else:
            query_vec_np = self._to_numpy(queries)

        da_list, scores = self._find_batched(
            query_vec_np, search_field=search_field, limit=limit, **kwargs
        )
        if (
            len(da_list) > 0
            and isinstance(da_list[0], List)
            and not isinstance(da_list[0], DocList)
        ):
            da_list = [self._dict_list_to_docarray(docs) for docs in da_list]

        return FindResultBatched(documents=da_list, scores=scores)  # type: ignore

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        """
        Conducts a batched search on the index.

        :param queries: The queries to search.
        :param limit: The maximum number of results to return for each query.
        :param search_field: The field to search the queries.
        :return: Search results.
        """

        self._collection.load()

        results = self._collection.search(
            data=queries,
            anns_field=self._field_name,
            param=self._db_config.search_params,
            limit=limit,
            expr=None,
            output_fields=["serialized"],
            consistency_level=self._db_config.consistency_level,
        )

        self._collection.release()

        documents, scores = zip(
            *[self._docs_from_find_response(result) for result in results]
        )

        return _FindResultBatched(
            documents=list(documents),
            scores=list(scores),
        )

    def execute_query(self, query: Any, *args, **kwargs) -> Any:
        """
        Executes a hybrid query on the index.

        :param query: Query to execute on the index.
        :return: Query results.
        """
        components: Dict[str, List[Dict[str, Any]]] = {}
        for component, value in query:
            if component not in components:
                components[component] = []
            components[component].append(value)

        if (
            len(components) != 2
            or len(components.get('find', [])) != 1
            or len(components.get('filter', [])) != 1
        ):
            raise ValueError(
                'The query must contain exactly one "find" and "filter" components.'
            )

        expr = components['filter'][0]['filter_query']
        query = components['find'][0]['query']
        limit = (
            components['find'][0].get('limit')
            or components['filter'][0].get('limit')
            or 10
        )
        docs, scores = self._hybrid_search(
            query=query,
            expr=expr,
            search_field=self._field_name,
            limit=limit,
        )
        if isinstance(docs, List) and not isinstance(docs, DocList):
            docs = self._dict_list_to_docarray(docs)

        return FindResult(documents=docs, scores=scores)

    def _docs_from_query_response(self, result: Sequence[Dict]) -> DocList[Any]:
        return DocList[self._schema](  # type: ignore
            [
                self._schema.from_base64(  # type: ignore
                    result[i]["serialized"], **self._db_config.serialize_config
                )
                for i in range(len(result))
            ]
        )

    def _docs_from_find_response(self, result: Hits) -> _FindResult:
        scores: NdArray = NdArray._docarray_from_native(
            np.array([hit.score for hit in result])
        )

        return _FindResult(
            documents=DocList[self.out_schema](  # type: ignore
                [
                    self.out_schema.from_base64(
                        hit.entity.get('serialized'), **self._db_config.serialize_config
                    )
                    for hit in result
                ]
            ),
            scores=scores,
        )

    def _always_true_expr(self, primary_key: str) -> str:
        """
        Returns a Milvus expression that is always true, thus allowing for the retrieval of all entries in a Collection.
        Assumes that the primary key is of type DataType.VARCHAR

        :param primary_key: the name of the primary key
        :return: a Milvus expression that is always true for that primary key
        """
        return f'({primary_key} in ["1"]) or ({primary_key} not in ["1"])'

    def _map_embedding(self, embedding: AnyTensor) -> np.ndarray:
        """
        Milvus exclusively supports one-dimensional vectors. If multi-dimensional
        vectors are provided, they will be automatically flattened to ensure compatibility.

        :param embedding: The original raw embedding, which can be in the form of a TensorFlow or PyTorch tensor.
        :return embedding: A one-dimensional numpy array representing the flattened version of the original embedding.
        """
        if embedding is None:
            raise ValueError(
                "Embedding is None. Each document must have a valid embedding."
            )

        embedding = self._to_numpy(embedding)
        if embedding.ndim > 1:
            embedding = np.asarray(embedding).squeeze()  # type: ignore

        return embedding

    def _doc_exists(self, doc_id: str) -> bool:
        result = self._collection.query(
            expr="id in " + str([doc_id]),
            offset=0,
            output_fields=["serialized"],
        )

        return len(result) > 0
