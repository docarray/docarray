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


def test_filter(start_storage):
    import random
    import string

    elastic_da = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 2,
            'columns': {
                'A': 'str',
                'B': 'str',
                'V': 'str',
                'D': 'str',
                'E': 'str',
                'F': 'str',
                'G': 'str',
            },
        },
    )

    def ran():
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    def ran_size():
        sizes = ['S', 'M', 'L', 'XL']
        return sizes[random.randint(0, len(sizes) - 1)]

    def ran_type():
        types = ['A', 'B', 'C', 'D']
        return types[random.randint(0, len(types) - 1)]

    def ran_stype():
        stypes = ['SA', 'SB', 'SC', 'SD']
        return stypes[random.randint(0, len(stypes) - 1)]

    docs = DocumentArray(
        [
            Document(
                id=f'r{i}',
                embedding=np.random.rand(2),
                tags={
                    'A': ran(),
                    'B': ran_stype(),
                    'C': ran_size(),
                    'D': ran_type(),
                    'E': ran(),
                    'F': ran_type(),
                    'G': f'G{i}',
                },
            )
            for i in range(50)
        ]
    )

    with elastic_da:
        elastic_da.extend(docs)

    res = elastic_da.find(query=Document(embedding=docs[0].embedding))
    assert len(res) > 0
    assert res[0][0].tags['G'] == 'G0'
    filter_ = {'match': {'G': 'G3'}}

    res = elastic_da.find(filter=filter_)
    assert len(res) > 0
    assert res[0].tags['G'] == 'G3'

    res = elastic_da.find(query=Document(embedding=docs[0].embedding), filter=filter_)
    assert len(res) > 0
    assert res[0][0].tags['G'] == 'G3'
