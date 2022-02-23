from typing import overload, Optional, Union, Dict, List, Tuple, Callable, TYPE_CHECKING
from ...math.helper import top_k, minmax_normalize, update_rows_x_mat_best
from ...score import NamedScore
from ...math import ndarray
import numpy as np


if TYPE_CHECKING:
    from ...types import T, ArrayType

    from ... import Document, DocumentArray


class FindMixin:
    """ A mixin that provides find functionality to DocumentArrays  """

    @overload
    def find(self: 'T', query: 'ArrayType', **kwargs):
        ...

    @overload
    def find(self: 'T', query: Union['Document', 'DocumentArray'], **kwargs):
        ...

    @overload
    def find(self: 'T', query: Dict, **kwargs):
        ...

    @overload
    def find(self: 'T', query: str, **kwargs):
        ...

    def find(
        self: 'T',
        query: Union['DocumentArray', 'Document', 'ArrayType'],
        limit: Optional[Union[int, float]] = 20,
        exclude_self: bool = False,
        only_id: bool = False,
        **kwargs,
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns approximate nearest neighbors given an input query.

        :param query: the input query to search by
        :param limit: the maximum number of matches, when not given defaults to 20.
        :param exclude_self: if set, Documents in results with same ``id`` as the query values will not be
                        considered as matches. This is only applied when the input query is Document or DocumentArray.
        :param only_id: if set, then returning matches will only contain ``id``
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """

        from ... import Document, DocumentArray

        if limit is not None:
            if limit <= 0:
                raise ValueError(f'`limit` must be larger than 0, receiving {limit}')
            else:
                limit = int(limit)

        if isinstance(query, (DocumentArray, Document)):
            _limit = (
                len(self) if limit is None else (limit + (1 if exclude_self else 0))
            )

            if isinstance(query, Document):
                query = DocumentArray(query)

            result = self._find(
                query.embeddings,
                limit=_limit,
                only_id=only_id,
                **kwargs,
            )
            if exclude_self:
                for i, q in enumerate(query):
                    matches = result[i].traverse_flat(
                        'r', filter_fn=lambda d: d.id != q.id
                    )
                    result[i] = matches[:limit] if _limit > limit else matches
        else:
            _limit = len(self) if limit is None else limit

            try:
                _, _ = ndarray.get_array_type(query)
                n_rows, n_dim = ndarray.get_array_rows(query)
            except TypeError:
                raise TypeError(
                    f'The find method of {self.__class__.__name__} does not support the type of query: {type(query)}'
                )

            # Ensure query embedding to have the correct shape
            if n_dim != 2:
                query = query.reshape((n_rows, -1))

            result = self._find(query, limit=_limit, only_id=only_id, **kwargs)

        if len(result) == 1:
            return result[0]
        else:
            return result

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
        only_id: bool = False,
        use_scipy: bool = False,
        device: str = 'cpu',
        num_worker: Optional[int] = 1,
        **kwargs,
    ) -> List['DocumentArray']:
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
        :param only_id: if set, then returning matches will only contain ``id``
        :param use_scipy: if set, use ``scipy`` as the computation backend. Note, ``scipy`` does not support distance
            on sparse matrix.
        :param device: the computational device for ``.search()``, can be either `cpu` or `cuda`.
        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.

                .. note::
                    This argument is only effective when ``batch_size`` is set.

        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """

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
                from ...math.distance import cdist as _cdist

                cdist = lambda *x: _cdist(*x, device=device)
        else:
            raise TypeError(
                f'metric must be either string or a 2-arity function, received: {metric!r}'
            )

        metric_name = metric_name or (metric.__name__ if callable(metric) else metric)

        if batch_size:
            dist, idx = self._search_online(
                query, cdist, limit, normalization, metric_name, batch_size, num_worker
            )
        else:
            dist, idx = self._search(query, cdist, limit, normalization, metric_name)

        from ... import Document, DocumentArray

        result = []
        for _ids, _dists in zip(idx, dist):
            matches = DocumentArray()
            for _id, _dist in zip(_ids, _dists):
                # Note, when match self with other, or both of them share the same Document
                # we might have recursive matches .
                # checkout https://github.com/jina-ai/jina/issues/3034
                if only_id:
                    d = Document(id=self[_id].id)
                else:
                    d = Document(self[int(_id)], copy=True)  # type: Document

                # to prevent self-reference and override on matches
                d.pop('matches')

                d.scores[metric_name] = NamedScore(value=_dist)
                matches.append(d)
                if len(matches) >= limit:
                    break
            result.append(matches)

        return result

    def _search(self, query: 'ArrayType', cdist, limit, normalization, metric_name):
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

    def _search_online(
        self,
        query,
        cdist,
        limit,
        normalization,
        metric_name,
        batch_size,
        num_worker,
    ):
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
