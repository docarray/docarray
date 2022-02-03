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

    def _find_similar_vectors(self, q, limit=10):
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

    def find(self, query, limit=10):
        """

        Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1`

        https://www.semi.technology/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
        """
        if query.ndim == 1:
            return self._find_similar_vectors(query, limit=limit)
        elif query.ndim == 2:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
