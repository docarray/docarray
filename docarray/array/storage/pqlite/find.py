from typing import (
    Union,
    TYPE_CHECKING,
    List,
)

if TYPE_CHECKING:
    import numpy as np
    from .... import DocumentArray


class FindMixin:
    def _find_similar_vectors(self, q: 'np.ndarray', limit=10):

        """
        if q.ndim == 1:
            input = DocumentArray(Document(embedding=q))
        else:
            input = DocumentArray(Document(embedding=q_k) for q_k in q)
        docs = self._pqlite.search(input, limit=limit)
        return DocumentArray(docs)
        """

        if q.ndim == 1:
            q = q.reshape((1, -1))

        _, list_of_docs = self._pqlite._search_documents(q, limit=limit)

        if len(list_of_docs) == 1:
            # this is a single DocumentArray
            return list_of_docs[0]
        else:
            # this is a list of DocumentArrays
            return list_of_docs

    def find(
        self, query: 'np.ndarray', limit: int = 10
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be stored in Weaviate. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.

        Note: Weaviate returns `certainty` values. To get cosine similarities one needs to use `cosine_sim = 2*certainty - 1` as explained here:
                  https://www.semi.technology/developers/weaviate/current/more-resources/faq.html#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty
        """

        return self._find_similar_vectors(query, limit=limit)
