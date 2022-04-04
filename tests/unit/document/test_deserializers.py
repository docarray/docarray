import json
import os

import numpy as np
from PIL import Image

from docarray import Document
from docarray.dataclasses.deserializers import (
    audio_deserializer,
    image_deserializer,
    json_deserializer,
    text_deserializer,
)

cur_dir = os.path.dirname(os.path.abspath(__file__))


IMAGE_URI = os.path.join(cur_dir, 'toydata/test.png')


def test_image_deserializer():
    doc = Document()
    doc._metadata['image_type'] = 'uri'
    doc._metadata['image_uri'] = 'image_uri'

    assert image_deserializer('attribute_name', doc) == 'image_uri'

    doc._metadata['image_type'] = 'ndarray'
    im = Image.open(IMAGE_URI)
    doc.tensor = np.asarray(im)

    assert np.all(image_deserializer('attribute_name', doc) == np.asarray(im))

    doc._metadata['image_type'] = 'PIL'

    assert np.all(
        image_deserializer('attribute_name', doc) == Image.fromarray(np.asarray(im))
    )


def test_text_deserializer():
    doc = Document(text='text')

    assert text_deserializer('attriute_name', doc) == 'text'


def test_json_deserializer():
    doc = Document()
    doc._metadata['json_type'] = ''
    doc.tags['attribute_name'] = 'attribute'
    assert json_deserializer('attribute_name', doc) == 'attribute'

    doc._metadata['json_type'] = 'str'

    assert json_deserializer('attribute_name', doc) == json.dumps('attribute')


def test_audio_deserializer():
    doc = Document()
    ad = Image.open(IMAGE_URI)
    doc.tensor = np.asarray(ad)

    assert np.all(
        audio_deserializer('attribute_name', doc) == Image.fromarray(np.asarray(ad))
    )
