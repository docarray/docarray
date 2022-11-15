import numpy as np

from docarray.array import DocumentArray
from docarray.document import BaseDocument
from docarray.typing import Tensor


def test_get_bukl_attributes():
    class Mmdoc(BaseDocument):
        text: str
        tensor: Tensor

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da._get_documents_attribute('tensor')

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da._get_documents_attribute('text')

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'
