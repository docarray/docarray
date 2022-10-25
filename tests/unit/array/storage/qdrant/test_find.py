import numpy as np
import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'search_params,expected',
    [
        ({'hnsw_ef': 64}, 64),
        (None, None),
    ],
)
def test_success_find_with_added_kwargs(
    search_params, expected, start_storage, monkeypatch
):
    nrof_docs = 10

    qdrant_doc = DocumentArray(
        storage='qdrant',
        config={
            'n_dim': 3,
        },
    )

    with qdrant_doc:
        qdrant_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ],
        )

    def _mock_search(*args, **kwargs):
        if expected:
            assert kwargs['search_params'].hnsw_ef == expected
        else:
            assert kwargs['search_params'] is None
        return []

    monkeypatch.setattr(qdrant_doc._client, 'search', _mock_search)

    np_query = np.array([2, 1, 3])

    qdrant_doc.find(np_query, limit=10, search_params=search_params)
