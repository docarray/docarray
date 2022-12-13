import warnings
from typing import List, Optional

import numpy as np


def _expand_if_single_axis(*matrices: 'np.array') -> List['np.array']:
    """Expands arrays that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be expanded
    :return: List of the input matrices,
        where single axis matrices are expanded at dim 0.
    """
    expanded = []
    for m in matrices:
        if len(m.shape) == 1:
            expanded.append(np.expand_dims(m, axis=0))
        else:
            expanded.append(m)
    return expanded


def _expand_if_scalar(arr: 'np.array') -> 'np.array':
    if len(arr.shape) == 0:  # avoid scalar output
        arr = np.expand_dims(arr, axis=0)
    return arr


def cosine_sim(
    x_mat: 'np.ndarray',
    y_mat: 'np.ndarray',
    eps: float = 1e-7,
    device: Optional[str] = None,
) -> 'np.ndarray':
    """Pairwise cosine distances between all vectors in x_mat and y_mat.

    :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param eps: a small jitter to avoid divde by zero
    :param device: Not supported for this backend
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        cosine distances.
        The index [i_x, i_y] contains the cosine distance between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        warnings.warn('`device` is not supported for numpy operations')

    x_mat, y_mat = _expand_if_single_axis(x_mat, y_mat)

    sims = np.clip(
        (np.dot(x_mat, y_mat.T) + eps)
        / (
            np.outer(np.linalg.norm(x_mat, axis=1), np.linalg.norm(y_mat, axis=1)) + eps
        ),
        -1,
        1,
    ).squeeze()
    return _expand_if_scalar(sims)


def sqeuclidean_dist(
    x_mat: 'np.ndarray',
    y_mat: 'np.ndarray',
    eps: float = 1e-7,
    device: Optional[str] = None,
) -> 'np.ndarray':
    """Pairwise Squared Euclidian distances between all vectors in x_mat and y_mat.

    :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param eps: a small jitter to avoid divde by zero
    :param eps: Negative results (due to numerical inaccuracies) no smaller than
        -eps will be returned as 0.
    :param device: Not supported for this backend
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        Squared Euclidian distances.
        The index [i_x, i_y] contains the cosine Squared Euclidian between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        warnings.warn('`device` is not supported for numpy operations')

    x_mat, y_mat = _expand_if_single_axis(x_mat, y_mat)

    dists = (
        np.sum(y_mat**2, axis=1)
        + np.sum(x_mat**2, axis=1)[:, np.newaxis]
        - 2 * np.dot(x_mat, y_mat.T)
    ).squeeze()

    # remove numerical artifacts
    dists = np.where(np.logical_and(dists < 0, dists > -eps), 0, dists)
    return _expand_if_scalar(dists)


def euclidean_dist(
    x_mat: 'np.ndarray', y_mat: 'np.ndarray', device: Optional[str] = None
) -> 'np.ndarray':
    """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

    :param x_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: np.ndarray of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param eps: a small jitter to avoid divde by zero
    :param device: Not supported for this backend
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        euclidian distances.
        The index [i_x, i_y] contains the euclidian distance between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        warnings.warn('`device` is not supported for numpy operations')

    x_mat, y_mat = _expand_if_single_axis(x_mat, y_mat)

    return _expand_if_scalar(np.sqrt(sqeuclidean_dist(x_mat, y_mat)).squeeze())
