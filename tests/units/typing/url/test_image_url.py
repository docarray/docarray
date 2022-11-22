import os
import urllib

import numpy as np
import PIL
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import ImageUrl

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_DATA = os.path.join(CUR_DIR, '..', '..', '..', 'toydata', 'image-data')
IMAGE_PATHS = {
    'png': os.path.join(PATH_TO_IMAGE_DATA, 'so_good.png'),
    'jpg': os.path.join(PATH_TO_IMAGE_DATA, '05984.jpg'),
    'jpeg': os.path.join(PATH_TO_IMAGE_DATA, '05984-2.jpeg'),
}
REMOTE_JPG = (
    'https://upload.wikimedia.org/wikipedia/commons/8/80/'
    'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
)


def test_image_url():
    uri = parse_obj_as(ImageUrl, REMOTE_JPG)

    tensor = uri.load()

    assert isinstance(tensor, np.ndarray)


def test_proto_image_url():

    uri = parse_obj_as(ImageUrl, REMOTE_JPG)

    uri._to_node_protobuf()


def test_json_schema():
    schema_json_of(ImageUrl)


def test_dump_json():
    url = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')
    orjson_dumps(url)


@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('remote-jpg', REMOTE_JPG),
    ],
)
def test_load(image_format, path_to_img):
    url = parse_obj_as(ImageUrl, path_to_img)
    tensor = url.load()
    assert isinstance(tensor, np.ndarray)


@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('remote-jpg', REMOTE_JPG),
    ],
)
@pytest.mark.parametrize('width,height', [(224, None), (None, 224), (224, 224)])
def test_load_width_height(image_format, path_to_img, width, height):
    url = parse_obj_as(ImageUrl, path_to_img)
    tensor = url.load(width=width, height=height)
    assert isinstance(tensor, np.ndarray)

    shape = tensor.shape
    if width:
        assert shape[1] == width
    if height:
        assert shape[0] == height


@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('remote-jpg', REMOTE_JPG),
    ],
)
@pytest.mark.parametrize(
    'axis_layout',
    [
        ('H', 'W', 'C'),
        ('H', 'C', 'W'),
        ('C', 'H', 'W'),
        ('C', 'W', 'H'),
        ('W', 'C', 'H'),
        ('W', 'H', 'C'),
    ],
)
def test_load_channel_axis(image_format, path_to_img, axis_layout):
    sizes = {'H': 100, 'W': 200, 'C': 3}
    url = parse_obj_as(ImageUrl, path_to_img)
    tensor = url.load(axis_layout=axis_layout, height=sizes['H'], width=sizes['W'])
    assert isinstance(tensor, np.ndarray)

    shape = tensor.shape
    for axis, axis_name in enumerate(axis_layout):
        assert shape[axis] == sizes[axis_name]


def test_load_timeout():
    url = parse_obj_as(ImageUrl, REMOTE_JPG)
    with pytest.raises(urllib.error.URLError):
        _ = url.load(timeout=0.001)


@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('jpg', REMOTE_JPG),
    ],
)
def test_load_to_bytes(image_format, path_to_img):
    w, h = 224, 224
    url = parse_obj_as(ImageUrl, path_to_img)
    _bytes = url.load_to_bytes(width=w, height=h)
    assert isinstance(_bytes, bytes)
    img = PIL.Image.frombytes(mode='1', size=(w, h), data=_bytes)
    assert isinstance(img, PIL.Image.Image)


@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('jpg', REMOTE_JPG),
        ('illegal', 'illegal'),
        ('illegal', 'https://www.google.com'),
        ('illegal', 'my/local/text/file.txt'),
    ],
)
def test_validation(image_format, path_to_img):
    if image_format == 'illegal':
        with pytest.raises(ValueError):
            parse_obj_as(ImageUrl, path_to_img)
    else:
        url = parse_obj_as(ImageUrl, path_to_img)
        assert isinstance(url, ImageUrl)
        assert isinstance(url, str)
