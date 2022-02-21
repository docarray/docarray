from typing import List, TYPE_CHECKING

import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from docarray.types import ArrayType


class QdrantStorageHelper:
    @classmethod
    def embedding_to_array(
        cls, embedding: 'ArrayType', default_dim: int
    ) -> List[float]:
        if embedding is None:
            embedding = np.random.rand(default_dim).tolist()  # [0.0] * default_dim
        elif isinstance(embedding, scipy.sparse.spmatrix):
            embedding = embedding.toarray().tolist()
        else:
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding).tolist()

        return embedding
