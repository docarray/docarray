import numpy as np

from docarray.typing import Tensor
from docarray import Document


def test_set_tensor():

    class MyDocument(Document):
        tensor: Tensor

    d = MyDocument(tensor=np.zeros((3, 224, 224)))

    assert isinstance(d.tensor, Tensor)
    assert isinstance(d.tensor, np.ndarray)
    assert (d.tensor == np.zeros((3, 224, 224))).all()
