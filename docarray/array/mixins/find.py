import abc
from typing import overload, Optional, Union, Dict, List, Tuple, Callable, TYPE_CHECKING

import numpy as np

from ...math import ndarray
from ...score import NamedScore

if TYPE_CHECKING:
    from ...types import T, ArrayType

    from ... import Document, DocumentArray


class FindMixin:
    """A mixin that provides find functionality to DocumentArrays

    Subclass should override :meth:`._find` not :meth:`.find`.
    """

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
        query: Union['DocumentArray', 'Document', 'ArrayType', Dict],
        metric: Union[
            str, Callable[['ArrayType', 'ArrayType'], 'np.ndarray']
        ] = 'cosine',
        limit: Optional[Union[int, float]] = 20,
        metric_name: Optional[str] = None,
        exclude_self: bool = False,
        only_id: bool = False,
        **kwargs,
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns approximate nearest neighbors given an input query.

        :param query: the input query to search by
        :param limit: the maximum number of matches, when not given defaults to 20.
        :param metric_name: if provided, then match result will be marked with this string.
        :param metric: the distance metric.
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

        if isinstance(query, dict):
            return self._filter(query, limit=limit, only_id=only_id)

        _limit = len(self) if limit is None else (limit + (1 if exclude_self else 0))
        if isinstance(query, (DocumentArray, Document)):

            if isinstance(query, Document):
                query = DocumentArray(query)

            _query = query.embeddings
        else:
            _query = query

        _, _ = ndarray.get_array_type(_query)
        n_rows, n_dim = ndarray.get_array_rows(_query)

        # Ensure query embedding to have the correct shape
        if n_dim != 2:
            _query = _query.reshape((n_rows, -1))

        metric_name = metric_name or (metric.__name__ if callable(metric) else metric)

        kwargs.update(
            {
                'limit': _limit,
                'only_id': only_id,
                'metric': metric,
                'metric_name': metric_name,
            }
        )

        _result = self._find(
            _query,
            **kwargs,
        )

        result: List['DocumentArray']

        if isinstance(_result, list) and isinstance(_result[0], DocumentArray):
            # already auto-boxed by the storage backend, e.g. annlite
            result = _result
        elif (
            isinstance(_result, tuple)
            and isinstance(_result[0], np.ndarray)
            and isinstance(_result[1], np.ndarray)
        ):
            # do autobox for Tuple['np.ndarray', 'np.ndarray']
            dist, idx = _result
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
                    if len(matches) >= _limit:
                        break
                result.append(matches)
        else:
            raise TypeError(
                f'unsupported type `{type(_result)}` returned from `._find()`'
            )

        if exclude_self and isinstance(query, DocumentArray):
            for i, q in enumerate(query):
                matches = result[i].traverse_flat('r', filter_fn=lambda d: d.id != q.id)
                if limit and len(matches) > limit:
                    result[i] = matches[:limit]
                else:
                    result[i] = matches

        if len(result) == 1:
            return result[0]
        else:
            return result

    @abc.abstractmethod
    def _find(
        self, query: 'ArrayType', limit: int, **kwargs
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        raise NotImplementedError

    def _filter(
        self,
        query: Dict,
        limit: Optional[int] = None,
        only_id: bool = False,
    ) -> 'DocumentArray':
        from ... import DocumentArray
        from ..queryset import QueryParser

        parser = QueryParser(query)

        result = DocumentArray()
        limit = len(self) if (limit is None) else limit
        for d in self:
            if parser.evaluate(d):
                result.append(d if not only_id else Document(id=d.id))
                if len(result) == limit:
                    break

        return result
