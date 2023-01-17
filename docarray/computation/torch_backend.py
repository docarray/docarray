from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from docarray.computation.abstract_comp_backend import AbstractComputationalBackend


def _unsqueeze_if_single_axis(*matrices: torch.Tensor) -> List[torch.Tensor]:
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


def _usqueeze_if_scalar(t: torch.Tensor):
    if len(t.shape) == 0:  # avoid scalar output
        t = t.unsqueeze(0)
    return t


class TorchCompBackend(AbstractComputationalBackend[torch.Tensor]):
    """
    Computational backend for PyTorch.
    """

    @staticmethod
    def stack(
        tensors: Union[List['torch.Tensor'], Tuple['torch.Tensor']], dim: int = 0
    ) -> 'torch.Tensor':
        return torch.stack(tensors, dim=dim)

    @staticmethod
    def to_device(tensor: 'torch.Tensor', device: str) -> 'torch.Tensor':
        """Move the tensor to the specified device."""
        return tensor.to(device)

    @staticmethod
    def empty(shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape)

    @staticmethod
    def n_dim(array: 'torch.Tensor') -> int:
        return array.ndim

    @staticmethod
    def to_numpy(array: 'torch.Tensor') -> 'np.ndarray':
        return array.cpu().detach().numpy()

    @staticmethod
    def none_value() -> Any:
        """Provide a compatible value that represents None in torch."""
        return torch.tensor(float('nan'))

    @staticmethod
    def shape(tensor: 'torch.Tensor') -> Tuple[int, ...]:
        return tuple(tensor.shape)

    @staticmethod
    def reshape(tensor: 'torch.Tensor', shape: Tuple[int, ...]) -> 'torch.Tensor':

        """
        Gives a new shape to tensor without changing its data.

        :param tensor: tensor to be reshaped
        :param shape: the new shape
        :return: a tensor with the same data and number of elements as tensor
            but with the specified shape.
        """
        return tensor.reshape(shape)

    class Retrieval(AbstractComputationalBackend.Retrieval[torch.Tensor]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        def top_k(
            values: 'torch.Tensor',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['torch.Tensor', 'torch.Tensor']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag.

            :param values: Torch tensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: the computational device to use,
                can be either `cpu` or a `cuda` device.
            :return: Tuple containing the retrieved values, and their indices.
                Both ar of shape (n_queries, k)
            """
            if device is not None:
                values = values.to(device)
            if len(values.shape) <= 1:
                values = values.view(1, -1)
            len_values = values.shape[-1] if len(values.shape) > 1 else len(values)
            k = min(k, len_values)
            return torch.topk(
                input=values, k=k, largest=descending, sorted=True, dim=-1
            )

    class Metrics(AbstractComputationalBackend.Metrics[torch.Tensor]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        def cosine_sim(
            x_mat: torch.Tensor,
            y_mat: torch.Tensor,
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> torch.Tensor:
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for pytorch computations.
                Either 'cpu' or a 'cuda' device.
                If not provided, the devices of x_mat and y_mat are used.
            :return: torch Tensor  of shape (n_vectors, n_vectors) containing all
                pairwise cosine distances.
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

        @staticmethod
        def euclidean_dist(
            x_mat: torch.Tensor, y_mat: torch.Tensor, device: Optional[str] = None
        ) -> torch.Tensor:
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param device: the device to use for pytorch computations.
                Either 'cpu' or a 'cuda' device.
                If not provided, the devices of x_mat and y_mat are used.
            :return: torch Tensor  of shape (n_vectors, n_vectors) containing all
                pairwise euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            if device is not None:
                x_mat = x_mat.to(device)
                y_mat = y_mat.to(device)

            x_mat, y_mat = _unsqueeze_if_single_axis(x_mat, y_mat)

            dists = torch.cdist(x_mat, y_mat).squeeze()
            return _usqueeze_if_scalar(dists)

        @staticmethod
        def sqeuclidean_dist(
            x_mat: torch.Tensor, y_mat: torch.Tensor, device: Optional[str] = None
        ) -> torch.Tensor:
            """Pairwise Squared Euclidian distances between all vectors in
            x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for pytorch computations.
                Either 'cpu' or a 'cuda' device.
                If not provided, the devices of x_mat and y_mat are used.
            :return: torch Tensor of shape (n_vectors, n_vectors) containing all
                pairwise Squared Euclidian distances.
                The index [i_x, i_y] contains the cosine Squared Euclidian between
                x_mat[i_x] and y_mat[i_y].
            """
            if device is not None:
                x_mat = x_mat.to(device)
                y_mat = y_mat.to(device)

            x_mat, y_mat = _unsqueeze_if_single_axis(x_mat, y_mat)

            return _usqueeze_if_scalar((torch.cdist(x_mat, y_mat) ** 2).squeeze())
