import numpy as np

from docarray import Document
from docarray.typing import Embedding


def test_set_embedding():
    class MyDocument(Document):
        embedding: Embedding

    d = MyDocument(embedding=np.zeros((3, 224, 224)))

    assert isinstance(d.embedding, Embedding)
    assert isinstance(d.embedding, np.ndarray)
    assert (d.embedding == np.zeros((3, 224, 224))).all()
