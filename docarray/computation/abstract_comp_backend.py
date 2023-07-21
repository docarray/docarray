import typing
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypeVar, Union, Iterable

if TYPE_CHECKING:
    import numpy as np

# In practice all of the below will be the same type
TTensor = TypeVar('TTensor')
TTensorRetrieval = TypeVar('TTensorRetrieval', bound=Iterable)
TTensorMetrics = TypeVar('TTensorMetrics')


class AbstractComputationalBackend(ABC, typing.Generic[TTensor]):
    """
    Abstract base class for computational backends.
    Every supported tensor/ML framework (numpy, torch etc.) should define its own
    computational backend exposing common functionality expressed in that framework.
    That way, DocList can leverage native implementations from all frameworks.
    """

    @classmethod
    @abstractmethod
    def stack(
        cls, tensors: Union[List['TTensor'], Tuple['TTensor']], dim: int = 0
    ) -> 'TTensor':
        """
        Stack a list of tensors along a new axis.
        """
        ...

    @classmethod
    @abstractmethod
    def copy(cls, tensor: 'TTensor') -> 'TTensor':
        """return a copy of the tensor"""
        ...

    @classmethod
    @abstractmethod
    def n_dim(cls, array: 'TTensor') -> int:
        """
        Get the number of the array dimensions.
        """
        ...

    @classmethod
    @abstractmethod
    def squeeze(cls, tensor: 'TTensor') -> 'TTensor':
        """
        Returns a tensor with all the dimensions of tensor of size 1 removed.
        """
        ...

    @classmethod
    @abstractmethod
    def to_numpy(cls, array: 'TTensor') -> 'np.ndarray':
        """
        Convert array to np.ndarray.
        """
        ...

    @classmethod
    @abstractmethod
    def empty(
        cls,
        shape: Tuple[int, ...],
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> 'TTensor':
        ...

    @classmethod
    @abstractmethod
    def none_value(cls) -> typing.Any:
        """Provide a compatible value that represents None in the Tensor Backend."""
        ...

    @classmethod
    @abstractmethod
    def to_device(cls, tensor: 'TTensor', device: str) -> 'TTensor':
        """Move the tensor to the specified device."""
        ...

    @classmethod
    @abstractmethod
    def device(cls, tensor: 'TTensor') -> Optional[str]:
        """Return device on which the tensor is allocated."""
        ...

    @classmethod
    @abstractmethod
    def shape(cls, tensor: 'TTensor') -> Tuple[int, ...]:
        """Get shape of tensor"""
        ...

    @classmethod
    @abstractmethod
    def reshape(cls, tensor: 'TTensor', shape: Tuple[int, ...]) -> 'TTensor':
        """
        Gives a new shape to tensor without changing its data.

        :param tensor: tensor to be reshaped
        :param shape: the new shape
        :return: a tensor with the same data and number of elements as tensor
            but with the specified shape.
        """
        ...

    @classmethod
    @abstractmethod
    def detach(cls, tensor: 'TTensor') -> 'TTensor':
        """
        Returns the tensor detached from its current graph.

        :param tensor: tensor to be detached
        :return: a detached tensor with the same data.
        """
        ...

    @classmethod
    @abstractmethod
    def dtype(cls, tensor: 'TTensor') -> Any:
        """Get the data type of the tensor."""
        ...

    @classmethod
    @abstractmethod
    def isnan(cls, tensor: 'TTensor') -> 'TTensor':
        """Check element-wise for nan and return result as a boolean array"""
        ...

    @classmethod
    @abstractmethod
    def minmax_normalize(
        cls,
        tensor: 'TTensor',
        t_range: Tuple = (0, 1),
        x_range: Optional[Tuple] = None,
        eps: float = 1e-7,
    ) -> 'TTensor':
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
        ...

    @classmethod
    @abstractmethod
    def equal(cls, tensor1: 'TTensor', tensor2: 'TTensor') -> bool:
        """
        Check if two tensors are equal.

        :param tensor1: the first tensor
        :param tensor2: the second tensor
        :return: True if two tensors are equal, False otherwise.
            If one or more of the inputs is not a tensor of this framework, return False.
        """
        ...

    class Retrieval(ABC, typing.Generic[TTensorRetrieval]):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        @abstractmethod
        def top_k(
            values: 'TTensorRetrieval',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['TTensorRetrieval', 'TTensorRetrieval']:
            """
            Retrieves the top k smallest values in `values`,
            and returns them alongside their indices in the input `values`.
            Can also be used to retrieve the top k largest values,
            by setting the `descending` flag to True.

            :param values: Tensor of values to rank.
                Should be of shape (n_queries, n_values_per_query).
                Inputs of shape (n_values_per_query,) will be expanded
                to (1, n_values_per_query).
            :param k: number of values to retrieve
            :param descending: retrieve largest values instead of smallest values
            :param device: the computational device to use.
            :return: Tuple containing the retrieved values, and their indices.
                Both ar of shape (n_queries, k)
            """
            ...

    class Metrics(ABC, typing.Generic[TTensorMetrics]):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        @abstractmethod
        def cosine_sim(
            x_mat: 'TTensorMetrics',
            y_mat: 'TTensorMetrics',
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> 'TTensorMetrics':
            """Pairwise cosine similarities between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param eps: a small jitter to avoid divde by zero
            :param device: the device to use for computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor  of shape (n_vectors, n_vectors) containing all pairwise
                cosine distances.
                The index [i_x, i_y] contains the cosine distance between
                x_mat[i_x] and y_mat[i_y].
            """
            ...

        @staticmethod
        @abstractmethod
        def euclidean_dist(
            x_mat: 'TTensorMetrics',
            y_mat: 'TTensorMetrics',
            device: Optional[str] = None,
        ) -> 'TTensorMetrics':
            """Pairwise Euclidian distances between all vectors in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            ...

        @staticmethod
        @abstractmethod
        def sqeuclidean_dist(
            x_mat: 'TTensorMetrics',
            y_mat: 'TTensorMetrics',
            device: Optional[str] = None,
        ) -> 'TTensorMetrics':
            """Pairwise Squared Euclidian distances between all vectors
                in x_mat and y_mat.

            :param x_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param y_mat: tensor of shape (n_vectors, n_dim), where n_vectors is the
                number of vectors and n_dim is the number of dimensions of each
                example.
            :param device: the device to use for pytorch computations.
                If not provided, the devices of x_mat and y_mat are used.
            :return: Tensor of shape (n_vectors, n_vectors) containing all pairwise
                euclidian distances.
                The index [i_x, i_y] contains the euclidian distance between
                x_mat[i_x] and y_mat[i_y].
            """
            ...
