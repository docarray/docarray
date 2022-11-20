import numpy as np

from docarray import Image


def test_image():

    image = Image(url='http://jina.ai')

    image.tensor = image.url.load()

    assert isinstance(image.tensor, np.ndarray)
