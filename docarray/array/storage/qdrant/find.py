from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, TypeVar, Union

from docarray import Document, DocumentArray
from docarray.math import ndarray
from docarray.score import NamedScore
from qdrant_client.http import models
from qdrant_client.http.models.models import Distance

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import tensorflow
    import torch
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

    def _find_similar_vectors(
        self,
        q: 'QdrantArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        search_params: Optional[Dict] = None,
        **kwargs,
    ):
        query_vector = self._map_embedding(q)

        search_result = self.client.search(
            self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            search_params=None
            if not search_params
            else models.SearchParams(**search_params),
            limit=limit,
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

    def _find(
        self,
        query: 'QdrantArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        search_params: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be used in Qdrant.
        :param limit: number of retrieved items
        :param filter: filter query used for pre-filtering
        :param search_params: additional parameters of the search


        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """

        num_rows, _ = ndarray.get_array_rows(query)

        if num_rows == 1:
            return [
                self._find_similar_vectors(
                    query, limit=limit, filter=filter, search_params=search_params
                )
            ]
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(
                    q, limit=limit, filter=filter, search_params=search_params
                )
                closest_docs.append(da)
            return closest_docs

    def _find_with_filter(
        self, filter: Optional[Dict], limit: Optional[Union[int, float]] = 10
    ):
        list_of_points, _offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(**filter),
            with_payload=True,
            limit=limit,
        )
        da = DocumentArray()
        for result in list_of_points[:limit]:
            doc = Document.from_base64(
                result.payload['_serialized'], **self.serialize_config
            )
            da.append(doc)
        return da

    def _filter(
        self, filter: Optional[Dict], limit: Optional[Union[int, float]] = 10
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given filter (`Qdrant` filter)..
        :param limit: number of retrieved items
        :param filter: filter query used for filtering.
        For more information: https://docarray.jina.ai/advanced/document-store/qdrant/#qdrant
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """

        return self._find_with_filter(filter, limit=limit)
