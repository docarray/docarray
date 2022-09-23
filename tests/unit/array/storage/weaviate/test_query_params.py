from docarray import Document, DocumentArray
import numpy as np


def test_certainty_filter(start_storage):
    nrof_docs = 100
    da = DocumentArray(
        storage="weaviate", config={"name": "Test", "host": "localhost", "port": 8080}
    )

    with da:
        da.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ],
        )

    results = da.find(
        DocumentArray([Document(embedding=[1, 2, 3])]),
        query_params={"certainty": 0.9}
    )

    #,
    #query_params = {"certainty": 0.1}
    print(len(results[0]))
    for res in results[0]:
        print(res[:, 'embedding'])
        print(res[:, 'scores'])
        print(res[:, 'scores'][0]['weaviate_certainty'])
        assert True
        #assert res[:, 'tags__lastUpdateTimeUnix'][0] is not None
