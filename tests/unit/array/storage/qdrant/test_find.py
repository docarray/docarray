from docarray import Document, DocumentArray
import numpy as np


def test_success_find_with_added_kwargs(start_storage):
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

    np_query = np.array([2, 1, 3])

    qdrant_doc.find(np_query, limit=10)
    qdrant_doc.find(np_query, limit=10, search_params={"hnsw_ef": 64})
