# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import urllib

import numpy as np
import PIL
import pytest
from PIL import Image
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import ImageUrl
from docarray.typing.url.mimetypes import (
    OBJ_MIMETYPE,
    AUDIO_MIMETYPE,
    VIDEO_MIMETYPE,
    IMAGE_MIMETYPE,
    TEXT_MIMETYPE,
)
from tests import TOYDATA_DIR

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


@pytest.mark.slow
@pytest.mark.internet
def test_image_url():
    uri = parse_obj_as(ImageUrl, REMOTE_JPG)

    tensor = uri.load()

    assert isinstance(tensor, np.ndarray)


@pytest.mark.proto
def test_proto_image_url():
    uri = parse_obj_as(ImageUrl, REMOTE_JPG)

    uri._to_node_protobuf()


def test_json_schema():
    schema_json_of(ImageUrl)


def test_dump_json():
    url = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')
    orjson_dumps(url)


@pytest.mark.slow
@pytest.mark.internet
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


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'image_format,path_to_img',
    [
        ('png', IMAGE_PATHS['png']),
        ('jpg', IMAGE_PATHS['jpg']),
        ('jpeg', IMAGE_PATHS['jpeg']),
        ('remote-jpg', REMOTE_JPG),
    ],
)
def test_load_pil(image_format, path_to_img):
    url = parse_obj_as(ImageUrl, path_to_img)
    img = url.load_pil()
    assert isinstance(img, PIL.Image.Image)


@pytest.mark.slow
@pytest.mark.internet
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


@pytest.mark.slow
@pytest.mark.internet
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


@pytest.mark.internet
def test_load_timeout():
    url = parse_obj_as(ImageUrl, REMOTE_JPG)
    with pytest.raises(urllib.error.URLError):
        _ = url.load(timeout=0.001)


@pytest.mark.slow
@pytest.mark.internet
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
    url = parse_obj_as(ImageUrl, path_to_img)
    _bytes = url.load_bytes()
    assert isinstance(_bytes, bytes)
    img = Image.frombytes(mode='1', size=(224, 224), data=_bytes)
    assert isinstance(img, Image.Image)


@pytest.mark.parametrize(
    'path_to_img',
    [*IMAGE_PATHS.values(), REMOTE_JPG],
)
def test_validation(path_to_img):
    url = parse_obj_as(ImageUrl, path_to_img)
    assert isinstance(url, ImageUrl)
    assert isinstance(url, str)


@pytest.mark.parametrize(
    'file_type, file_source',
    [
        (IMAGE_MIMETYPE, IMAGE_PATHS['png']),
        (IMAGE_MIMETYPE, IMAGE_PATHS['jpg']),
        (IMAGE_MIMETYPE, IMAGE_PATHS['jpeg']),
        (IMAGE_MIMETYPE, REMOTE_JPG),
        (AUDIO_MIMETYPE, os.path.join(TOYDATA_DIR, 'hello.mp3')),
        (AUDIO_MIMETYPE, os.path.join(TOYDATA_DIR, 'hello.wav')),
        (VIDEO_MIMETYPE, os.path.join(TOYDATA_DIR, 'mov_bbb.mp4')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'test' 'test.html')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'test' 'test.md')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'penal_colony.txt')),
        (OBJ_MIMETYPE, os.path.join(TOYDATA_DIR, 'test.glb')),
    ],
)
def test_file_validation(file_type, file_source):
    if file_type != ImageUrl.mime_type():
        with pytest.raises(ValueError):
            parse_obj_as(ImageUrl, file_source)
    else:
        parse_obj_as(ImageUrl, file_source)
