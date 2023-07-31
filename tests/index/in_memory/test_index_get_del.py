import numpy as np

from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


def test_update_payload():
    docs = DocList[SimpleDoc](
        [SimpleDoc(embedding=np.random.rand(128), text=f'hey {i}') for i in range(100)]
    )
    index = InMemoryExactNNIndex[SimpleDoc]()
    index.index(docs)

    assert index.num_docs() == 100

    for doc in docs:
        doc.text += '_changed'

    index.index(docs)
    assert index.num_docs() == 100

    res = index.find(query=docs[0], search_field='embedding', limit=100)
    assert len(res.documents) == 100
    for doc in res.documents:
        assert '_changed' in doc.text


def test_update_embedding():
    docs = DocList[SimpleDoc](
        [SimpleDoc(embedding=np.random.rand(128), text=f'hey {i}') for i in range(100)]
    )
    index = InMemoryExactNNIndex[SimpleDoc]()
    index.index(docs)
    assert index.num_docs() == 100

    new_tensor = np.random.rand(128)
    docs[0].embedding = new_tensor

    index.index(docs[0])
    assert index.num_docs() == 100

    res = index.find(query=docs[0], search_field='embedding', limit=100)
    assert len(res.documents) == 100
    found = False
    for doc in res.documents:
        if doc.id == docs[0].id:
            found = True
            assert (doc.embedding == new_tensor).all()
    assert found
