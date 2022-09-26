from docarray import Document, DocumentArray
import numpy as np


def find_random(da, target_certainty):
    return da.find(
        DocumentArray([Document(embedding=np.random.randint(10, size=10))]),
        query_params={'certainty': target_certainty},
    )[0]


def test_certainty_filter(start_storage):
    nrof_docs = 100
    target_certainty = 0.98
    da = DocumentArray(
        storage='weaviate', config={'n_dim': 10}
    )

    with da:
        da.extend(
            [
                Document(embedding=np.random.randint(10, size=10))
                for i in range(1, nrof_docs)
            ],
        )

    results = []
    while len(results) == 0:
        results = find_random(da, target_certainty)

    for res in results:
        assert res.scores['weaviate_certainty'].value >= target_certainty
