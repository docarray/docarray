import numpy as np

from docarray import Image
from docarray.typing import Tensor


def test_image():

    image = Image(uri='http://jina.ai')

    image.tensor = image.uri.load()

    assert isinstance(image.tensor, np.ndarray)
