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
        query: Union[
            'DocumentArray', 'Document', 'ArrayType', Dict, str, List[str], None
        ] = None,
        metric: Union[
            str, Callable[['ArrayType', 'ArrayType'], 'np.ndarray']
        ] = 'cosine',
        limit: Optional[Union[int, float]] = 20,
        metric_name: Optional[str] = None,
        exclude_self: bool = False,
        filter: Optional[Dict] = None,
        only_id: bool = False,
        index: str = 'text',
        **kwargs,
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns matching Documents given an input query.
        If the query is a `DocumentArray`, `Document` or `ArrayType`, exhaustive or approximate nearest neighbor search
        will be performed depending on whether the storage backend supports ANN. Furthermore, if filter is not None,
        pre-filtering will be applied along with vector search.
        If the query is a `dict` object or, query is None and filter is not None, Documents will be filtered and all
        matching Documents that match the filter will be returned. In this case, query (if it's dict) or filter will be
        used for filtering. The object must follow the backend-specific filter format if the backend supports filtering
        or DocArray's query language format. In the latter case, filtering will be applied in the client side not the
        backend side.
        If the query is a string or list of strings, a search by text will be performed if the backend supports
        indexing and searching text fields. If not, a `NotImplementedError` will be raised.

        :param query: the input query to search by
        :param limit: the maximum number of matches, when not given defaults to 20.
        :param metric_name: if provided, then match result will be marked with this string.
        :param metric: the distance metric.
        :param exclude_self: if set, Documents in results with same ``id`` as the query values will not be
                        considered as matches. This is only applied when the input query is Document or DocumentArray.
        :param filter: filter query used for pre-filtering or filtering
        :param only_id: if set, then returning matches will only contain ``id``
        :param index: if the query is a string, text search will be performed on the `index` field, otherwise, this
                      parameter is ignored. By default, the Document `text` attribute will be used for search,
                      otherwise the tag field specified by `index` will be used. You can only use this parameter if the
                      storage backend supports searching by text.
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """

        from ... import Document, DocumentArray

        if isinstance(query, dict):
            if filter is None:
                return self._filter(query)
            else:
                raise ValueError(
                    'filter and query cannot be both dict type, set only one for filtering'
                )
        elif query is None:
            if isinstance(filter, dict):
                return self._filter(filter)
            else:
                raise ValueError('filter must be dict when query is None')
        elif isinstance(query, str) or (
            isinstance(query, list) and isinstance(query[0], str)
        ):
            if filter is not None:
                raise ValueError('cannot use filter with text search')
            result = self._find_by_text(query, index=index, limit=limit, **kwargs)
            if isinstance(query, str):
                return result[0]
            else:
                return result

        # for all the rest, vector search will be performed
        elif isinstance(query, (DocumentArray, Document)):

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
            filter=filter,
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

        # ensures query=np.array([1,2,3]) returns DocumentArray not list with 1 DocumentArray
        if n_dim == 1:
            result = result[0]

        return result

    @abc.abstractmethod
    def _find(
        self, query: 'ArrayType', limit: int, filter: Optional[Dict] = None, **kwargs
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

    def _find_by_text(self, query: Union[str, List[str]], index: str = 'text'):
        raise NotImplementedError('Search by text is not supported')
