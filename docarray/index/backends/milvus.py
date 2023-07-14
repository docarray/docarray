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

import numpy as np

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.index.backends.helper import _execute_find_and_filter_query
from docarray.typing import AnyTensor, NdArray
from docarray.typing.id import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import (
    _FindResult,
    _FindResultBatched,
    FindResult,
    FindResultBatched,
)
from docarray.array.any_array import AnyDocArray

if TYPE_CHECKING:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
        Hits,
    )
else:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
        Hits,
    )

ID_VARCHAR_LEN = 1024
SERIALIZED_VARCHAR_LEN = 65_535  # Maximum length that Milvus allows for a VARCHAR field

TSchema = TypeVar('TSchema', bound=BaseDoc)


class MilvusDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        """Initialize MilvusDocumentIndex"""
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

        self._validate_columns()
        self._field_name = self._get_vector_field_name()
        self._create_collection_name()
        self._collection = self._init_index()
        self._build_index()
        self._collection.load()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of MilvusDocumentIndex."""

        collection_name: Optional[str] = None
        collection_description: str = ""
        host: str = "localhost"
        port: int = 19530
        user: Optional[str] = ""
        password: Optional[str] = ""
        token: Optional[str] = ""
        index_type: str = "IVF_FLAT"
        index_metric: str = "L2"
        index_params: Dict = field(default_factory=lambda: {"nlist": 1024})
        consistency_level: str = 'Session'
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

        return None

    def _init_index(self) -> Collection:
        """
        This function initializes or retrieves a Milvus collection with a specified schema,
        storing documents as serialized data and using the document's ID as the collection's ID
        , while inheriting other schema properties from the indexer's schema.

        !!! note
            Milvus framework currently only supports a single vector column, and only one vector
            column can store in the schema (others are stored in the serialized data)
        """

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
                    max_length=ID_VARCHAR_LEN,
                ),
            ]
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
                            {'dim': info.n_dim or info.config.get('dim')}
                            if info.db_type == DataType.FLOAT_VECTOR
                            else {}
                        ),
                    )
                    for column_name, info in self._column_infos.items()
                    if column_name != 'id'
                    and not (
                        info.db_type == DataType.FLOAT_VECTOR
                        and column_name != self._field_name
                    )  # Only store one vector field in column
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
        """
        This function generates a unique and sanitized name for the collection,
        , ensuring a unique identifier is used if the user does not specify a
        collection name.
        """
        if self._db_config.collection_name is None:
            id = uuid.uuid4().hex
            self._db_config.collection_name = f"{self.__class__.__name__}__" + id

        self._db_config.collection_name = ''.join(
            re.findall('[a-zA-Z0-9_]', self._db_config.collection_name)
        )

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
        return self._db_config.collection_name

    def _build_index(self):
        """
        Sets up an index configuration for a specific column index, which is
        required by the Milvus backend.
        """

        index = {
            "index_type": self._db_config.index_type,
            "metric_type": self._db_config.index_metric,
            "params": self._db_config.index_params,
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
        docs_validated = self._validate_docs(docs)
        self._update_subindex_data(docs_validated)

        docs = self._validate_docs(docs)
        entities: List[List[Any]] = [[] for _ in range(len(self._collection.schema))]

        for i in range(len(docs)):
            entities[0].append(docs[i].to_base64(**self._db_config.serialize_config))
            entity_index = 1
            for column_name, info in self._column_infos.items():
                column_value = self._get_values_by_column([docs[i]], column_name)[0]
                if info.db_type == DataType.FLOAT_VECTOR:
                    if column_name != self._field_name:
                        continue
                    column_value = self._map_embedding(column_value)

                entities[entity_index].append(column_value)
                entity_index += 1

        self._collection.insert(entities)
        self._collection.flush()
        self._logger.info(f"{len(docs)} documents has been indexed")

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

        result = self._collection.query(
            expr="id in " + str([id for id in doc_ids]),
            offset=0,
            output_fields=["serialized"],
            consistency_level=self._db_config.consistency_level,
        )

        self._collection.release()

        return self._docs_from_query_response(result)

    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        self._collection.load()
        self._collection.delete(
            expr="id in " + str([id for id in doc_ids]),
            consistency_level=self._db_config.consistency_level,
        )

        self._logger.info(f"{len(doc_ids)} documents has been deleted")

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> Union[DocList, List[Dict]]:
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
        return [
            self._filter(filter_query=query, limit=limit) for query in filter_queries
        ]

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        raise NotImplementedError

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
        self._collection.load()

        results = self._collection.search(
            data=[query],
            anns_field=search_field,
            param=self._db_config.search_params,
            limit=limit,
            offset=0,
            expr=None,
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
                if issubclass(self._schema._get_field_type(fields[0]), AnyDocArray):  # type: ignore
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
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )
        find_res = _execute_find_and_filter_query(
            doc_index=self,
            query=query,
        )
        return find_res

    def _docs_from_query_response(self, result: Sequence[Dict]) -> Sequence[TSchema]:
        return DocList[self._schema](
            [
                self._schema.from_base64(
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
            documents=DocList[self._schema](
                [
                    self._schema.from_base64(
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

    def _map_embedding(self, embedding: Optional[AnyTensor]) -> Optional[AnyTensor]:
        """
        Milvus exclusively supports one-dimensional vectors. If multi-dimensional
        vectors are provided, they will be automatically flattened to ensure compatibility.

        :param embedding: The original raw embedding, which can be in the form of a TensorFlow or PyTorch tensor.
        :return embedding: A one-dimensional numpy array representing the flattened version of the original embedding.
        """

        if embedding is not None:
            embedding = self._to_numpy(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()
        else:
            embedding = np.zeros(self._db_config.n_dim)
        return embedding

    def __contains__(self, item) -> bool:
        result = self._collection.query(
            expr="id in " + str([item.id]),
            offset=0,
            output_fields=["serialized"],
        )

        return len(result) > 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collection.release()
        self._loaded = False
