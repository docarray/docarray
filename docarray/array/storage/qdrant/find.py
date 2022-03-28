from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
)

from qdrant_openapi_client.models.models import Distance

from .... import Document, DocumentArray
from ....math import ndarray
from ....score import NamedScore

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
    @abstractmethod
    def client(self) -> 'QdrantClient':
        raise NotImplementedError()

    @property
    @abstractmethod
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def serialize_config(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def distance(self) -> 'Distance':
        raise NotImplementedError()

    def _find_similar_vectors(self, q: 'QdrantArrayType', limit=10):
        query_vector = self._map_embedding(q)

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
                hit.payload['_serialized'].value[0], **self.serialize_config
            )
            doc.scores[f'{self.distance.lower()}_similarity'] = NamedScore(
                value=hit.score
            )
            docs.append(doc)

        return DocumentArray(docs)

    def _find(
        self, query: 'QdrantArrayType', limit: int = 10, **kwargs
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be used in Qdrant.
        :param limit: number of retrieved items


        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """

        num_rows, _ = ndarray.get_array_rows(query)

        if num_rows == 1:
            return [self._find_similar_vectors(query, limit=limit)]
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
