from typing import (
    Union,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
)

from .helper import QdrantStorageHelper
from .... import Document, DocumentArray
from ....math import ndarray
from ....score import NamedScore
from qdrant_openapi_client.models.models import Distance

if TYPE_CHECKING:
    import tensorflow
    import torch
    import numpy as np
    from qdrant_client import QdrantClient

    QdrantArrayType = TypeVar(
        'QdrantArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin:
    @property
    def client(self) -> 'QdrantClient':
        raise NotImplementedError()

    @property
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    def serialize_config(self) -> dict:
        raise NotImplementedError()

    @property
    def distance(self) -> 'Distance':
        raise NotImplementedError()

    def _find_similar_vectors(self, q: 'QdrantArrayType', limit=10):
        query_vector = QdrantStorageHelper.embedding_to_array(q, default_dim=0)

        search_result = self.client.search(
            self.collection_name,
            query_vector=query_vector,
            query_filter=None,
            search_params=None,
            top=limit,
            append_payload=['_serialized'],
        )

        docs = []

        for hit in search_result:
            doc = Document.from_base64(
                hit.payload['_serialized'], **self.serialize_config
            )
            doc.scores[f'{self.distance.lower()}_similarity'] = NamedScore(
                value=hit.score
            )
            docs.append(doc)

        return DocumentArray(docs)

    def find(
        self, query: 'QdrantArrayType', limit: int = 10
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be used in Qdrant.
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """

        num_rows, _ = ndarray.get_array_rows(query)

        if num_rows == 1:
            return self._find_similar_vectors(query, limit=limit)
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
