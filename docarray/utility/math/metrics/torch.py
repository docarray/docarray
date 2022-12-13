from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:  # pragma: no cover
    from torch import tensor


def _unsqueeze_if_single_axis(*matrices: 'tensor') -> List['tensor']:
    """Unsqueezes tensors that only have one axis, at dim 0.
    This ensures that all outputs can be treated as matrices, not vectors.

    :param matrices: Matrices to be unsqueezed
    :return: List of the input matrices,
        where single axis matrices are unsqueezed at dim 0.
    """
    unsqueezed = []
    for m in matrices:
        if len(m.shape) == 1:
            unsqueezed.append(m.unsqueeze(0))
        else:
            unsqueezed.append(m)
    return unsqueezed


def _usqueeze_if_scalar(t: 'tensor'):
    if len(t.shape) == 0:  # avoid scalar output
        t = t.unsqueeze(0)
    return t


def cosine_sim(
    x_mat: 'tensor', y_mat: 'tensor', eps: float = 1e-7, device: Optional[str] = None
) -> 'tensor':
    """Pairwise cosine similarities between all vectors in x_mat and y_mat.

    :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param eps: a small jitter to avoid divde by zero
    :param device: the device to use for pytorch computations.
        Either 'cpu' or a 'cuda' device.
        If not provided, the devices of x_mat and y_mat are used.
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        cosine distances.
        The index [i_x, i_y] contains the cosine distance between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        x_mat = x_mat.to(device)
        y_mat = y_mat.to(device)

    x_mat, y_mat = _unsqueeze_if_single_axis(x_mat, y_mat)

    a_n, b_n = x_mat.norm(dim=1)[:, None], y_mat.norm(dim=1)[:, None]
    a_norm = x_mat / torch.clamp(a_n, min=eps)
    b_norm = y_mat / torch.clamp(b_n, min=eps)
    sims = torch.mm(a_norm, b_norm.transpose(0, 1)).squeeze()
    return _usqueeze_if_scalar(sims)


def euclidean_dist(
    x_mat: 'tensor', y_mat: 'tensor', device: Optional[str] = None
) -> 'tensor':
    """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

    :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param device: the device to use for pytorch computations.
        Either 'cpu' or a 'cuda' device.
        If not provided, the devices of x_mat and y_mat are used.
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        euclidian distances.
        The index [i_x, i_y] contains the euclidian distance between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        x_mat = x_mat.to(device)
        y_mat = y_mat.to(device)

    x_mat, y_mat = _unsqueeze_if_single_axis(x_mat, y_mat)

    dists = torch.cdist(x_mat, y_mat).squeeze()
    return _usqueeze_if_scalar(dists)


def sqeuclidean_dist(
    x_mat: 'tensor', y_mat: 'tensor', device: Optional[str] = None
) -> 'tensor':
    """Pairwise Squared Euclidian distances between all vectors in x_mat and y_mat.

    :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
        number of vectors and n_dim is the number of dimensions of each example.
    :param eps: a small jitter to avoid divde by zero
    :param device: the device to use for pytorch computations.
        Either 'cpu' or a 'cuda' device.
        If not provided, the devices of x_mat and y_mat are used.
    :return: np.ndarray  of shape (n_vectors, n_vectors) containing all pairwise
        Squared Euclidian distances.
        The index [i_x, i_y] contains the cosine Squared Euclidian between
        x_mat[i_x] and y_mat[i_y].
    """
    if device is not None:
        x_mat = x_mat.to(device)
        y_mat = y_mat.to(device)

    x_mat, y_mat = _unsqueeze_if_single_axis(x_mat, y_mat)

    return _usqueeze_if_scalar((torch.cdist(x_mat, y_mat) ** 2).squeeze())
