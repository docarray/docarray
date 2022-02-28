from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
)

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.helper import EPSILON
from ....math.ndarray import to_numpy_array
from ....score import NamedScore

if TYPE_CHECKING:
    import tensorflow
    import torch

    WeaviateArrayType = TypeVar(
        'WeaviateArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin:
    def _find_similar_vectors(self, query: 'WeaviateArrayType', limit=10):
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        query_dict = {'vector': query}
        results = (
            self._client.query.get(
                self._class_name,
                ['_serialized', '_additional {certainty}', '_additional {id}'],
            )
            .with_limit(limit)
            .with_near_vector(query_dict)
            .do()
        )
        docs = []

        # The serialized document is stored in results['data']['Get'][self._class_name]
        for result in results.get('data', {}).get('Get', {}).get(self._class_name, []):
            doc = Document.from_base64(result['_serialized'], **self._serialize_config)
            certainty = result['_additional']['certainty']

            doc.scores['weaviate_certainty'] = NamedScore(value=certainty)

            if certainty is None:
                doc.scores['cosine_similarity'] = NamedScore(value=None)
            else:
                doc.scores['cosine_similarity'] = NamedScore(value=2 * certainty - 1)

            doc.tags = {
                'wid': result['_additional']['id'],
            }
            docs.append(doc)

        return DocumentArray(docs)

    def _find(
        self, query: 'WeaviateArrayType', limit: int = 10, **kwargs
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be stored in Weaviate. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.

        Note: Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1` as explained here:
                  https://www.semi.technology/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
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
