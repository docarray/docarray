from typing import Optional, Union, Tuple, Callable, TYPE_CHECKING, Dict

import numpy as np

from ....math import ndarray
from ....math.helper import top_k, minmax_normalize, update_rows_x_mat_best

if TYPE_CHECKING:
    from ....typing import T, ArrayType

    from .... import DocumentArray


class FindMixin:
    """A mixin that provides find functionality to DocumentArrays"""

    def _find(
        self: 'T',
        query: 'ArrayType',
        metric: Union[
            str, Callable[['ArrayType', 'ArrayType'], 'np.ndarray']
        ] = 'cosine',
        limit: Optional[Union[int, float]] = 20,
        normalization: Optional[Tuple[float, float]] = None,
        metric_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_scipy: bool = False,
        device: str = 'cpu',
        num_worker: Optional[int] = 1,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        """Returns approximate nearest neighbors given a batch of input queries.

        :param query: the query embeddings to search
        :param metric: the distance metric.
        :param limit: the maximum number of matches, when not given defaults to 20.
        :param normalization: a tuple [a, b] to be used with min-max normalization,
                                the min distance will be rescaled to `a`, the max distance will be rescaled to `b`
                                all values will be rescaled into range `[a, b]`.
        :param metric_name: if provided, then match result will be marked with this string.
        :param batch_size: if provided, then ``self.embeddings`` is loaded in batches, where each of them is at most ``batch_size``
            elements. When `self.embeddings` is big, this can significantly speedup the computation.
        :param use_scipy: if set, use ``scipy`` as the computation backend. Note, ``scipy`` does not support distance
            on sparse matrix.
        :param device: the computational device for ``.search()``, can be either `cpu` or `cuda`.
        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.

                .. note::
                    This argument is only effective when ``batch_size`` is set.
        :param filter: filter query used for pre-filtering
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        if filter is not None:
            raise ValueError(
                'Filtered vector search is not supported for In-Memory backend'
            )

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError(
                    f'`batch_size` must be larger than 0, receiving {batch_size}'
                )
            else:
                batch_size = int(batch_size)

        if callable(metric):
            cdist = metric
        elif isinstance(metric, str):
            if use_scipy:
                from scipy.spatial.distance import cdist as cdist
            else:
                from ....math.distance import cdist as _cdist

                cdist = lambda *x: _cdist(*x, device=device)
        else:
            raise TypeError(
                f'metric must be either string or a 2-arity function, received: {metric!r}'
            )

        metric_name = metric_name or (metric.__name__ if callable(metric) else metric)

        if batch_size:
            return self._find_nn_online(
                query, cdist, limit, normalization, metric_name, batch_size, num_worker
            )
        else:
            return self._find_nn(query, cdist, limit, normalization, metric_name)

    def _find_nn(
        self, query: 'ArrayType', cdist, limit, normalization, metric_name
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        """
        :param query: the query embeddings to search by.
        :param cdist: the distance metric
        :param limit: the maximum number of matches, when not given
                      all Documents in `darray` are considered as matches
        :param normalization: a tuple [a, b] to be used with min-max normalization,
                                the min distance will be rescaled to `a`, the max distance will be rescaled to `b`
                                all values will be rescaled into range `[a, b]`.
        :param metric_name: if provided, then match result will be marked with this string.
        :return: distances and indices
        """

        dists = cdist(query, self.embeddings, metric_name)
        dist, idx = top_k(dists, min(limit, len(self)), descending=False)
        if isinstance(normalization, (tuple, list)) and normalization is not None:
            # normalization bound uses original distance not the top-k trimmed distance
            min_d = np.min(dists, axis=-1, keepdims=True)
            max_d = np.max(dists, axis=-1, keepdims=True)
            dist = minmax_normalize(dist, normalization, (min_d, max_d))

        return dist, idx

    def _find_nn_online(
        self,
        query,
        cdist,
        limit,
        normalization,
        metric_name,
        batch_size,
        num_worker,
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        """

        :param query: the query embeddings to search by.
        :param cdist: the distance metric
        :param limit: the maximum number of matches, when not given
                      all Documents in `another` are considered as matches
        :param normalization: a tuple [a, b] to be used with min-max normalization,
                              the min distance will be rescaled to `a`, the max distance will be rescaled to `b`
                              all values will be rescaled into range `[a, b]`.
        :param batch_size: length of the chunks loaded into memory from darray.
        :param metric_name: if provided, then match result will be marked with this string.
        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :return: distances and indices
        """
        n_q, _ = ndarray.get_array_rows(query)

        idx = 0
        top_dists = np.inf * np.ones((n_q, limit))
        top_inds = np.zeros((n_q, limit), dtype=int)

        def _get_dist(da: 'DocumentArray'):
            distances = cdist(query, da.embeddings, metric_name)
            dists, inds = top_k(distances, limit, descending=False)

            if isinstance(normalization, (tuple, list)) and normalization is not None:
                dists = minmax_normalize(dists, normalization)

            return dists, inds, len(da)

        if num_worker is None or num_worker > 1:
            # notice that all most all computations (regardless the framework) are conducted in C
            # hence there is no worry on Python GIL and the backend can be safely put to `thread` to
            # save unnecessary data passing. This in fact gives a huge boost on the performance.
            _gen = self.map_batch(
                _get_dist,
                batch_size=batch_size,
                backend='thread',
                num_worker=num_worker,
            )
        else:
            _gen = (_get_dist(b) for b in self.batch(batch_size=batch_size))

        for (dists, inds, _bs) in _gen:
            inds += idx
            idx += _bs
            top_dists, top_inds = update_rows_x_mat_best(
                top_dists, top_inds, dists, inds, limit
            )

        # sort final the final `top_dists` and `top_inds` per row
        permutation = np.argsort(top_dists, axis=1)
        dist = np.take_along_axis(top_dists, permutation, axis=1)
        idx = np.take_along_axis(top_inds, permutation, axis=1)

        return dist, idx
