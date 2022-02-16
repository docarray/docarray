from typing import (
    Union,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
)

from .... import Document, DocumentArray
from ....math import ndarray
from ....score import NamedScore

if TYPE_CHECKING:
    import tensorflow
    import torch
    import numpy as np

    WeaviateArrayType = TypeVar(
        'WeaviateArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin:
    def _find_similar_vectors(self, q: 'WeaviateArrayType', limit=10):
        query_dict = {'vector': q}
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
            doc.scores['cosine_similarity'] = NamedScore(value=2 * certainty - 1)
            doc.tags = {
                'wid': result['_additional']['id'],
            }
            docs.append(doc)

        return DocumentArray(docs)

    def search(self, query: 'DocumentArray', limit: int = 10) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: the DocumentArray to search by their embeddings.
        :param limit: number of retrieved items

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.

        Note: Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1` as explained here:
                  https://www.semi.technology/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
        """

        result = []
        for q in query:
            matches = self._find_similar_vectors(
                ndarray.to_numpy_array(q.embedding), limit=limit
            )
            result.append(matches)
        return result
