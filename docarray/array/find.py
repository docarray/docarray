from docarray import DocumentArray
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import ArrayType


def top_k(
    values: 'np.ndarray', k: int, descending: bool = False
) -> Tuple['np.ndarray', 'np.ndarray']:
    """Finds values and indices of the k largest entries for the last dimension.

    :param values: array of distances
    :param k: number of values to retrieve
    :param descending: find top k biggest values
    :return: indices and distances
    """
    if descending:
        values = -values

    if k >= values.shape[1]:
        idx = values.argsort(axis=1)[:, :k]
        values = np.take_along_axis(values, idx, axis=1)
    else:
        idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
        values = np.take_along_axis(values, idx_ps, axis=1)
        idx_fs = values.argsort(axis=1)
        idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
        values = np.take_along_axis(values, idx_fs, axis=1)

    if descending:
        values = -values

    return values, idx


def cosine(x_mat: 'np.ndarray', y_mat: 'np.ndarray', eps: float = 1e-7) -> 'np.ndarray':
    """Cosine distance between each row in x_mat and each row in y_mat.

    :param x_mat: np.ndarray with ndim=2
    :param y_mat: np.ndarray with ndim=2
    :param eps: a small jitter to avoid divde by zero
    :return: np.ndarray  with ndim=2
    """
    return 1 - np.clip(
        (np.dot(x_mat, y_mat.T) + eps)
        / (
            np.outer(np.linalg.norm(x_mat, axis=1), np.linalg.norm(y_mat, axis=1)) + eps
        ),
        -1,
        1,
    )


def find(
    query_embedding: Optional['ArrayType'] = None,
    index: DocumentArray = DocumentArray([]),
    limit: Optional[Union[int, float]] = 20,
    embedding_field: Optional[str] = 'embedding',
) -> DocumentArray:
    """
    :param query_embeddings:
    Tensor with embeddings.
    :param index:
    :return: A list of DocumentArrays
    @param embedding_field:
    @param limit:
    @param query_embedding:
    @type index: object
    """
    distance_list = getattr(index, embedding_field)
    y_mat = np.asarray(distance_list)
    dists = cosine(query_embedding, y_mat)
    dist, idx = top_k(dists, min(limit, len(index)), descending=False)

    return DocumentArray(dist)
