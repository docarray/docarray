from tokenize import String
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

if TYPE_CHECKING:
    import tensorflow
    import torch

    WeaviateArrayType = TypeVar(
        'WeaviateArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )

options_without_cross_refs = [
    '_docarray_id',
    'blob',
    'tensor',
    'text',
    'granularity',
    'adjacency',
    'parent_id',
    'weight',
    'uri',
    'modality',
    'mime_type',
    'offset',
    'location']

options_with_cross_refs = ['chunks', 'matches']


class FindMixin:

    def collect_results_from_weaviate(self, result, include_cosine):
        """Returns a Doc based on Weaviate results

        :param result: the result from Weaviate
        :param include_cosine: should cosine distance be included in the results?
        """

        doc = Document(id=result['_docarray_id'])

        if include_cosine == True:
            certainty = result['_additional']['certainty']
            doc.scores['weaviate_certainty'] = NamedScore(value=certainty)
            if certainty is None:
                doc.scores['cosine_similarity'] = NamedScore(value=None)
            else:
                doc.scores['cosine_similarity'] = NamedScore(value=2 * certainty - 1)

        # populate all the values for non-cross-ref values
        for opt in options_without_cross_refs:  
            if opt in result:
                setattr(doc, opt, result[opt])

        # populate all cross ref values
        add_array= {}
        for opt in options_with_cross_refs:
            add_array[opt] = []
            if opt in result and result[opt] != None:
                    for cref_doc in result[opt]:
                        add_array[opt].append(self._getitem(self._map_id(cref_doc['_docarray_id'])))
            setattr(doc, opt, add_array[opt])

        doc.tags['wid'] = result['_additional']['id']

        if len(result['_additional']['vector']) > 0:
            setattr(doc, 'embedding', result['_additional']['vector'])

        return doc
        

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

        _additional = ['id', 'certainty', 'vector']
        if additional:
            _additional = _additional + additional

        query_builder = (
            self._client.query.get(self._class_name, options_without_cross_refs)
            .with_additional(_additional)
            .with_limit(limit)
            .with_near_vector(query_dict)
        )

        if filter:
            query_builder = query_builder.with_where(filter)

        if sort:
            query_builder = query_builder.with_sort(sort)

        results = query_builder.do()

        docs = []
        if 'errors' in results:
            errors = '\n'.join(map(lambda error: error['message'], results['errors']))
            raise ValueError(
                f'find failed, please check your filter query. Errors: \n{errors}'
            )

        found_results = results.get('data', {}).get('Get', {}).get(self._class_name, [])

        # The serialized document is stored in results['data']['Get'][self._class_name]
        for result in found_results:
            docs.append(self.collect_results_from_weaviate(result, True))

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

        # if the id is specified, use this
        if 'id' in filter:
            filter = { 'path': 'id', 'operator': 'Equal', 'valueString': self._map_id(filter['id']) }
            limit = 1

        _additional = ['id', 'vector']
        if additional:
            _additional = _additional + additional

        query_builder = (
            self._client.query.get(self._class_name, options_without_cross_refs + 
                ['chunks { ... on ' + self._class_name + ' { _docarray_id } }',
                'matches { ... on ' + self._class_name + ' { _docarray_id } }']
            )
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
            docs.append(self.collect_results_from_weaviate(result, False))

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
