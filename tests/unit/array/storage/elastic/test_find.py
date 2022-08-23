from docarray import Document, DocumentArray
import numpy as np


def test_success_find_with_added_kwargs(start_storage, monkeypatch):
    nrof_docs = 1000
    num_candidates = 100

    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'distance': 'l2_norm',
            'index_name': 'test_success_find_with_added_kwargs',
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ],
        )

    def _mock_knn_search(**kwargs):
        assert kwargs['knn']['num_candidates'] == num_candidates

        return {'hits': {'hits': []}}

    monkeypatch.setattr(elastic_doc._client, 'knn_search', _mock_knn_search)

    np_query = np.array([2, 1, 3])

    elastic_doc.find(np_query, limit=10, num_candidates=num_candidates)
