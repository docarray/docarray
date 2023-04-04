import numpy as np

from docarray import BaseDoc
from docarray.typing import AnyEmbedding


def test_set_embedding():
    class MyDocument(BaseDoc):
        embedding: AnyEmbedding

    d = MyDocument(embedding=np.zeros((3, 224, 224)))

    assert isinstance(d.embedding, np.ndarray)
    assert (d.embedding == np.zeros((3, 224, 224))).all()
