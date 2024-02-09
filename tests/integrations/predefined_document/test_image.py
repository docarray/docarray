// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import ImageBytes
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

REMOTE_JPG = (
    'https://upload.wikimedia.org/wikipedia/commons/8/80/'
    'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
)

pytestmark = [pytest.mark.image]


@pytest.mark.slow
@pytest.mark.internet
def test_image():
    image = ImageDoc(url=REMOTE_JPG)

    image.tensor = image.url.load()

    assert isinstance(image.tensor, np.ndarray)


def test_image_str():
    image = parse_obj_as(ImageDoc, 'http://myurl.jpg')
    assert image.url == 'http://myurl.jpg'


def test_image_np():
    image = parse_obj_as(ImageDoc, np.zeros((10, 10, 3)))
    assert (image.tensor == np.zeros((10, 10, 3))).all()


def test_image_torch():
    image = parse_obj_as(ImageDoc, torch.zeros(10, 10, 3))
    assert (image.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.tensorflow
def test_image_tensorflow():
    image = ImageDoc(tensor=tf.zeros((10, 10, 3)))
    assert tnp.allclose(image.tensor.tensor, tf.zeros((10, 10, 3)))


def test_image_shortcut_doc():
    class MyDoc(BaseDoc):
        image: ImageDoc
        image2: ImageDoc
        image3: ImageDoc

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
    img = ImageDoc(url=REMOTE_JPG)
    img.bytes_ = img.url.load_bytes()
    assert isinstance(img.bytes_, ImageBytes)


@pytest.mark.slow
@pytest.mark.internet
def test_byte_from_tensor():
    img = ImageDoc(url=REMOTE_JPG)
    img.tensor = img.url.load()
    img.bytes_ = img.tensor.to_bytes()

    assert isinstance(img.bytes_, bytes)
    assert isinstance(img.bytes_, ImageBytes)
    assert len(img.bytes_) > 0
