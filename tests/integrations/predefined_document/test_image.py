import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import Image

try:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp
except (ImportError, TypeError):
    pass

REMOTE_JPG = (
    'https://upload.wikimedia.org/wikipedia/commons/8/80/'
    'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
)


@pytest.mark.slow
@pytest.mark.internet
def test_image():
    image = Image(url=REMOTE_JPG)

    image.tensor = image.url.load()

    assert isinstance(image.tensor, np.ndarray)


def test_image_str():
    image = parse_obj_as(Image, 'http://myurl.jpg')
    assert image.url == 'http://myurl.jpg'


def test_image_np():
    image = parse_obj_as(Image, np.zeros((10, 10, 3)))
    assert (image.tensor == np.zeros((10, 10, 3))).all()


def test_image_torch():
    image = parse_obj_as(Image, torch.zeros(10, 10, 3))
    assert (image.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.tensorflow
def test_image_tensorflow():
    image = Image(tensor=tf.zeros((10, 10, 3)))
    assert tnp.allclose(image.tensor.tensor, tf.zeros((10, 10, 3)))


def test_image_shortcut_doc():
    class MyDoc(BaseDocument):
        image: Image
        image2: Image
        image3: Image

    doc = MyDoc(
        image='http://myurl.jpg',
        image2=np.zeros((10, 10, 3)),
        image3=torch.zeros(10, 10, 3),
    )
    assert doc.image.url == 'http://myurl.jpg'
    assert (doc.image2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.image3.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.slow
@pytest.mark.internet
def test_byte():

    img = Image(url=REMOTE_JPG)
    img.bytes = img.url.load_bytes()


@pytest.mark.slow
@pytest.mark.internet
def test_byte_from_tensor():

    img = Image(url=REMOTE_JPG)
    img.tensor = img.url.load()
    img.bytes = img.tensor.to_bytes()

    assert isinstance(img.bytes, bytes)
    assert len(img.bytes) > 0
