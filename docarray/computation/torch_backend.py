from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from docarray.computation.abstract_comp_backend import AbstractComputationalBackend
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import torch
else:
    torch = import_library('torch', raise_error=True)


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


def _unsqueeze_if_scalar(t: torch.Tensor):
    if len(t.shape) == 0:  # avoid scalar output
        t = t.unsqueeze(0)
    return t


class TorchCompBackend(AbstractComputationalBackend[torch.Tensor]):
    """
    Computational backend for PyTorch.
    """

    @classmethod
    def stack(
        cls, tensors: Union[List['torch.Tensor'], Tuple['torch.Tensor']], dim: int = 0
    ) -> 'torch.Tensor':
        return torch.stack(tensors, dim=dim)

    @classmethod
    def copy(cls, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """return a copy/clone of the tensor"""
        return tensor.clone()

    @classmethod
    def to_device(cls, tensor: 'torch.Tensor', device: str) -> 'torch.Tensor':
        """Move the tensor to the specified device."""
        return tensor.to(device)

    @classmethod
    def device(cls, tensor: 'torch.Tensor') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        return str(tensor.device)

    @classmethod
    def empty(
        cls,
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> torch.Tensor:
        extra_param = {}
        if dtype is not None:
            extra_param['dtype'] = dtype
        if device is not None:
            extra_param['device'] = device

        return torch.empty(shape, **extra_param)

    @classmethod
    def n_dim(cls, array: 'torch.Tensor') -> int:
        return array.ndim

    @classmethod
    def squeeze(cls, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """
        Returns a tensor with all the dimensions of tensor of size 1 removed.
        """
        return torch.squeeze(tensor)

    @classmethod
    def to_numpy(cls, array: 'torch.Tensor') -> 'np.ndarray':
        return array.cpu().detach().numpy()

    @classmethod
    def none_value(
        cls,
    ) -> Any:
        """Provide a compatible value that represents None in torch."""
        return torch.tensor(float('nan'))

    @classmethod
    def shape(cls, tensor: 'torch.Tensor') -> Tuple[int, ...]:
        return tuple(tensor.shape)

    @classmethod
    def reshape(cls, tensor: 'torch.Tensor', shape: Tuple[int, ...]) -> 'torch.Tensor':
        """
        Gives a new shape to tensor without changing its data.

        :param tensor: tensor to be reshaped
        :param shape: the new shape
        :return: a tensor with the same data and number of elements as tensor
            but with the specified shape.
        """
        return tensor.reshape(shape)

    @classmethod
    def equal(cls, tensor1: 'torch.Tensor', tensor2: 'torch.Tensor') -> bool:
        """
        Check if two tensors are equal.

        :param tensor1: the first tensor
        :param tensor2: the second tensor
        :return: True if two tensors are equal, False otherwise.
            If one or more of the inputs is not a torch.Tensor, return False.
        """
        are_torch = isinstance(tensor1, torch.Tensor) and isinstance(
            tensor2, torch.Tensor
        )
        return are_torch and torch.equal(tensor1, tensor2)

    @classmethod
    def detach(cls, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        return tensor.detach()

    @classmethod
    def dtype(cls, tensor: 'torch.Tensor') -> torch.dtype:
        """Get the data type of the tensor."""
        return tensor.dtype

    @classmethod
    def isnan(cls, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Check element-wise for nan and return result as a boolean array"""
        return torch.isnan(tensor)

    @classmethod
    def minmax_normalize(
        cls,
        tensor: 'torch.Tensor',
        t_range: Tuple = (0, 1),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'torch.Tensor':
        """
        Normalize values in `tensor` into `t_range`.

        `tensor` can be a 1D array or a 2D array. When `tensor` is a 2D array, then
        normalization is row-based.

        !!! note

            - with `t_range=(0, 1)` will normalize the min-value of data to 0, max to 1;
            - with `t_range=(1, 0)` will normalize the min-value of data to 1, max value
              of the data to 0.

        :param tensor: the data to be normalized
        :param t_range: a tuple represents the target range.
        :param x_range: a tuple represents tensors range.
        :param eps: a small jitter to avoid divide by zero
        :return: normalized data in `t_range`
        """
        a, b = t_range

        min_d = (
            x_range[0] if x_range else torch.min(tensor, dim=-1, keepdim=True).values
        )
        max_d = (
            x_range[1] if x_range else torch.max(tensor, dim=-1, keepdim=True).values
        )
        r = (b - a) * (tensor - min_d) / (max_d - min_d + eps) + a

        normalized = torch.clip(r, *((a, b) if a < b else (b, a)))
        return normalized.to(tensor.dtype)

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
            return _unsqueeze_if_scalar(sims)

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
            return _unsqueeze_if_scalar(dists)

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

            return _unsqueeze_if_scalar((torch.cdist(x_mat, y_mat) ** 2).squeeze())
