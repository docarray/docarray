from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from docarray.typing.tensor.tensor import Tensor


class AbstractComputationalBackend(ABC):
    """
    Abstract base class for computational backends.
    Every supported tensor/ML framework (numpy, torch etc.) should define its own
    computational backend exposing common functionality expressed in that framework.
    That way, DocArray can leverage native implementations from all frameworks.
    """

    @staticmethod
    @abstractmethod
    def stack(
        tensors: Union[List['Tensor'], Tuple['Tensor']], dim: int = 0
    ) -> 'Tensor':
        """
        Stack a list of tensors along a new axis.
        """
        ...

    class Retrieval(ABC):
        """
        Abstract class for retrieval and ranking functionalities
        """

        @staticmethod
        @abstractmethod
        def top_k(
            values: 'Tensor',
            k: int,
            descending: bool = False,
            device: Optional[str] = None,
        ) -> Tuple['Tensor', 'Tensor']:
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

    class Metrics(ABC):
        """
        Abstract base class for metrics (distances and similarities).
        """

        @staticmethod
        @abstractmethod
        def cosine_sim(
            x_mat: 'Tensor',
            y_mat: 'Tensor',
            eps: float = 1e-7,
            device: Optional[str] = None,
        ) -> 'Tensor':
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
            x_mat: 'Tensor', y_mat: 'Tensor', device: Optional[str] = None
        ) -> 'Tensor':
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
            x_mat: 'Tensor', y_mat: 'Tensor', device: Optional[str] = None
        ) -> 'Tensor':
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
