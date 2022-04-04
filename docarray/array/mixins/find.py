import abc
from typing import overload, Optional, Union, Dict, List, Tuple, Callable, TYPE_CHECKING

import numpy as np

from ...math import ndarray
from ...score import NamedScore

if TYPE_CHECKING:
    from ...typing import T, ArrayType

    from ... import Document, DocumentArray


class FindMixin:
    """A mixin that provides find functionality to DocumentArrays

    Subclass should override :meth:`._find` not :meth:`.find`.
    """

    @overload
    def find(
        self: 'T',
        query: Union['Document', 'DocumentArray', 'ArrayType'],
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
        ...

    @overload
    def find(self: 'T', query: Dict, **kwargs) -> 'DocumentArray':
        """Find Documents that meet certain query language and return the result as a DocumentArray.

        The query language we provide now is following the
        [MongoDB](https://docs.mongodb.com/manual/reference/operator/query/) query language. For example::

            >>> docs.find({'text': {'$eq': 'hello'}})

            The above will return a `DocumentArray` in which each document has doc.text == 'hello'. And we also support
            placeholder format by using the following syntax::

            >>> docs.find({'text': {'$eq': '{tags__name}'}})

            will return a `DocumentArray` in which each document has doc.text == doc.tags['name'].

        Now, only the subset of MongoDB's query operators are supported:
            - `$eq` - Equal to (number, string)
            - `$ne` - Not equal to (number, string)
            - `$gt` - Greater than (number)
            - `$gte` - Greater than or equal to (number)
            - `$lt` - Less than (number)
            - `$lte` - Less than or equal to (number)
            - `$in` - Included in an array
            - `$nin` - Not included in an array
            - `$regex` - Match a specified regular expression
            - `$size` - The array/dict field is a specified size. $size does not accept ranges of values.
            - `$exists` - Matches documents that have the specified field. And empty string content is also cosidered as not exists.

        And the following boolean logic operators are supported:
            - `$and` - Join query clauses with a logical AND
            - `$or` - Join query clauses with a logical OR
            - `$not` - Inverts the effect of a query expression

        :param query: the query language in a dict object
        :return: selected Documents in a DocumentArray
        """
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

        if isinstance(query, dict):
            return self._filter(query)
        if isinstance(query, (DocumentArray, Document)):

            if isinstance(query, Document):
                query = DocumentArray(query)

            _query = query.embeddings
        else:
            _query = query

        if limit is not None:
            if limit <= 0:
                raise ValueError(f'`limit` must be larger than 0, receiving {limit}')
            else:
                limit = int(limit)

        _limit = len(self) if limit is None else (limit + (1 if exclude_self else 0))

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
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given query.

        :return: a `DocumentArray` containing the `Document` objects for matching with the query.
        """
        from ... import DocumentArray
        from ..queryset import QueryParser

        if query:
            parser = QueryParser(query)
            return DocumentArray(d for d in self if parser.evaluate(d))
        else:
            return self
