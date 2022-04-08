from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Union,
    Dict,
    Callable,
    Optional,
)

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.helper import EPSILON
from ....math.ndarray import to_numpy_array
from ....score import NamedScore
from ....array.mixins.find import FindMixin as BaseFindMixin

if TYPE_CHECKING:
    import tensorflow
    import torch
    from ....typing import T, ArrayType

    ElasticArrayType = TypeVar(
        'ElasticArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin(BaseFindMixin):
    def _find_similar_vectors(self, query: 'ElasticArrayType', limit=10):
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        resp = self._client.knn_search(
            index=self._config.index_name,
            knn={
                'field': 'embedding',
                'query_vector': query,
                'k': limit,
                'num_candidates': 10000,
            },
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            doc.embedding = result['_source']['embedding']
            da.append(doc)

        return da

    def _find_similar_documents_from_text(self, query: str, limit=10):
        """
        Return key-word matches for the input query

        :param query: text used for key-word search
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """

        resp = self._client.search(
            index=self._config.index_name,
            query={'match': {'text': query}},
            source=['id', 'blob', 'text'],
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits[:limit]:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            da.append(doc)

        return da

    def _find(
        self,
        query: Union['ElasticArrayType', str, List[str]],
        limit: int = 10,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries if the input is an 'ElasticArrayType'.
           Returns exact key-word search if the input is `str` or `List[str]` if the input.

        :param query: input supported to be stored in Elastic. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        if isinstance(query[0], str):
            search_method = self._find_similar_documents_from_text
            num_rows = len(query) if isinstance(query, list) else 1
        else:
            search_method = self._find_similar_vectors
            query = np.array(query)
            num_rows, n_dim = ndarray.get_array_rows(query)
            if n_dim != 2:
                query = query.reshape((num_rows, -1))

        if num_rows == 1:
            # if it is a list do query[0]
            return [search_method(query[0], limit=limit)]
        else:
            closest_docs = []
            for q in query:
                da = search_method(q, limit=limit)
                closest_docs.append(da)
            return closest_docs

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

        from .... import Document, DocumentArray

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
