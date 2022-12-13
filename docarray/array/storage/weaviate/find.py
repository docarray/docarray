from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Dict,
    Optional,
    Union,
)

import numpy as np

from docarray import Document, DocumentArray
from docarray.math import ndarray
from docarray.math.helper import EPSILON
from docarray.math.ndarray import to_numpy_array
from docarray.score import NamedScore

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow
    import torch

    WeaviateArrayType = TypeVar(
        'WeaviateArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin:
    def _find_similar_vectors(
        self,
        query: 'WeaviateArrayType',
        limit=10,
        filter: Optional[Dict] = None,
        additional: Optional[List] = None,
        sort: Optional[Union[Dict, List]] = None,
        query_params: Optional[Dict] = None,
    ):
        """Returns a subset of documents by the given vector.

        :param query: input supported to be stored in Weaviate. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items
        :param filter: the input filter to apply in each stored document
        :param additional: Optional Weaviate flags for meta data
        :param sort: sort parameters performed on matches performed on results
        :param query_params: additional parameters applied to the query outside of the where clause
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        query_dict = {'vector': query}

        if query_params:
            query_dict.update(query_params)

        _additional = ['id', 'distance']
        if additional:
            _additional = _additional + additional

        query_builder = (
            self._client.query.get(self._class_name, '_serialized')
            .with_additional(_additional)
            .with_limit(limit)
            .with_near_vector(query_dict)
        )

        if filter is not None:
            query_builder = query_builder.with_where(filter)

        if sort is not None:
            query_builder = query_builder.with_sort(sort)

        results = query_builder.do()

        if 'errors' in results:
            errors = '\n'.join(map(lambda error: error['message'], results['errors']))
            raise ValueError(
                f'find failed, please check your filter query. Errors: \n{errors}'
            )

        found_results = results.get('data', {}).get('Get', {}).get(self._class_name, [])

        # The serialized document is stored in results['data']['Get'][self._class_name]

        docs = []

        for result in found_results:
            doc = Document.from_base64(result['_serialized'], **self._serialize_config)

            distance = result['_additional']['distance']
            doc.scores['distance'] = NamedScore(value=distance)

            certainty = result['_additional'].get('certainty', None)
            if certainty is not None:
                doc.scores['weaviate_certainty'] = NamedScore(value=certainty)

            doc.tags['wid'] = result['_additional']['id']

            if additional:
                for add in additional:
                    doc.tags[f'{add}'] = result['_additional'][add]

            docs.append(doc)

        return DocumentArray(docs)

    def _filter(
        self,
        filter: Dict,
        limit: Optional[Union[int, float]] = 20,
        additional: Optional[List] = None,
        sort: Optional[Union[Dict, List]] = None,
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given filter (Weaviate `where` filter).

        :param filter: the input filter to apply in each stored document
        :param limit: number of retrieved items
        :param additional: Optional Weaviate flags for meta data
        :param sort: sort parameters performed on matches performed on results
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """
        if not filter:
            return self

        _additional = ['id']
        if additional:
            _additional = _additional + additional

        query_builder = (
            self._client.query.get(self._class_name, '_serialized')
            .with_additional(_additional)
            .with_where(filter)
            .with_limit(limit)
        )

        if sort:
            query_builder = query_builder.with_sort(sort)

        results = query_builder.do()

        docs = []
        if 'errors' in results:
            errors = '\n'.join(map(lambda error: error['message'], results['errors']))
            raise ValueError(
                f'filter failed, please check your filter query. Errors: \n{errors}'
            )

        found_results = results.get('data', {}).get('Get', {}).get(self._class_name, [])

        # The serialized document is stored in results['data']['Get'][self._class_name]
        for result in found_results:
            doc = Document.from_base64(result['_serialized'], **self._serialize_config)

            doc.tags['wid'] = result['_additional']['id']

            if additional:
                for add in additional:
                    doc.tags[f'{add}'] = result['_additional'][add]

            docs.append(doc)

        return DocumentArray(docs)

    def _find(
        self,
        query: 'WeaviateArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        additional: Optional[List] = None,
        sort: Optional[Union[Dict, List]] = None,
        query_params: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be stored in Weaviate. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items
        :param filter: filter query used for pre-filtering
        :param additional: Optional Weaviate flags for meta data
        :param sort: sort parameters performed on matches performed on results
        :param query_params: additional parameters applied to the query outside of the where clause

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.

        Note: Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1` as explained here:
                  https://weaviate.io/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
        """

        num_rows, _ = ndarray.get_array_rows(query)

        if num_rows == 1:
            return [
                self._find_similar_vectors(
                    query,
                    limit=limit,
                    additional=additional,
                    filter=filter,
                    sort=sort,
                    query_params=query_params,
                )
            ]
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(
                    q,
                    limit=limit,
                    additional=additional,
                    filter=filter,
                    sort=sort,
                    query_params=query_params,
                )
                closest_docs.append(da)
            return closest_docs
