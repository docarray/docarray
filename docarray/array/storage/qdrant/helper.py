from typing import List, TYPE_CHECKING

import numpy as np
import scipy.sparse
from qdrant_openapi_client.models.models import Distance

from docarray.math.helper import EPSILON

if TYPE_CHECKING:
    from docarray.types import ArrayType


class QdrantStorageHelper:
    @classmethod
    def embedding_to_array(
        cls, embedding: 'ArrayType', default_dim: int
    ) -> List[float]:
        if embedding is None:
            embedding = np.random.rand(default_dim)
        else:
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

        if embedding.ndim > 1:
            embedding = np.asarray(embedding).squeeze()

        if np.all(embedding == 0):
            embedding = embedding + EPSILON
        return embedding.tolist()


DISTANCES = {
    'cosine': Distance.COSINE,
    'euclidean': Distance.EUCLID,
    'dot': Distance.DOT,
}
