import numpy as np

from docarray import BaseDocument
from docarray.typing import NdArray


def test_set_tensor():
    class MyDocument(BaseDocument):
        tensor: NdArray

    d = MyDocument(tensor=np.zeros((3, 224, 224)))

    assert isinstance(d.tensor, NdArray)
    assert isinstance(d.tensor, np.ndarray)
    assert (d.tensor == np.zeros((3, 224, 224))).all()
