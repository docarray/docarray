from docarray import Document, DocumentArray
import numpy as np


def find_random(da, target_certainty):
    return da.find(
        DocumentArray([Document(embedding=np.random.randint(10, size=10))]),
        query_params={"certainty": target_certainty},
    )[0]


def test_certainty_filter(start_storage):
    nrof_docs = 100
    target_certainty = 0.98
    da = DocumentArray(
        storage="weaviate", config={"name": "Test", "host": "localhost", "port": 8080}
    )

    with da:
        da.extend(
            [
                Document(id=f"r{i}", embedding=np.random.randint(10, size=10))
                for i in range(1, nrof_docs)
            ],
        )

    while True:
        results = find_random(da, target_certainty)
        if len(results) > 0:
            break

    for res in results:
        assert res.scores["weaviate_certainty"].value >= target_certainty
