import numpy as np
from typing import (
    Union,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    Optional,
    List,
    Iterable,
    Tuple,
    overload,
)

from .... import Document, DocumentArray

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
    @overload
    def find(self, query: 'WeaviateArrayType'):
        ...

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
            doc = Document.from_base64(result['_serialized'], **self.serialize_config)
            certainty = result['_additional']['certainty']
            doc.tags = {
                'weaviate_certainty': certainty,
                'cosine_similarity': 2 * certainty - 1,
                'wid': result['_additional']['id'],
            }

            docs.append(doc)

        return DocumentArray(docs)

    def _check_single_query(self, query: 'WeaviateArrayType'):
        """Return whether query contains a single or multiple queries across all compatible Weaviate types

        Examples

        >>> _check_single_query([1,2,3])
        True
        >>> _check_single_query([[1,2,3],[4,5,6]])
        Flase
        >>> _check_single_query(np.array([1,2,3]))
        True
        >>> _check_single_query(np.array([[1,2,3], [4,5,6]]))
        False
        >>> _check_single_query(tf.constant([1,2,3]))
        True
        >>> _check_single_query(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
        False
        """
        if isinstance(query, (np.ndarray, torch.Tensor, tf.Tensor)):
            return True if query.ndim >= 2 else False
        if isinstance(query, list, tuple):
            return True if isinstance(q[0], (list, tuple)) else False

    def find(
        self, query: 'WeaviateArrayType', limit=10
    ) -> Union[DocumentArray, List[DocumentArray]]:
        """
        :param query: input supported to be stored in Weaviate. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.

        Note: Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1` as explained here:
                  https://www.semi.technology/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
        """
        if self._check_single_query(query):
            return self._find_similar_vectors(query, limit=limit)
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
