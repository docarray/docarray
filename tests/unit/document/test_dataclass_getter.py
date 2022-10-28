import os

import numpy as np
from PIL import Image

from docarray import Document
from docarray.dataclasses.getter import (
    audio_getter,
    image_getter,
    json_getter,
    text_getter,
    uri_getter,
)
from docarray.dataclasses.enums import DocumentMetadata, ImageType

cur_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_URI = os.path.join(cur_dir, 'toydata/test.png')


def test_image_deserializer():
    doc = Document()
    doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.URI
    doc.uri = 'image_uri'

    assert image_getter(doc) == 'image_uri'

    doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.NDARRAY
    im = Image.open(IMAGE_URI)
    doc.tensor = np.asarray(im)

    assert np.all(image_getter(doc) == np.asarray(im))

    doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.PIL

    assert np.all(image_getter(doc) == Image.fromarray(np.asarray(im)))


def test_text_deserializer():
    doc = Document(text='text')

    assert (
        text_getter(
            doc,
        )
        == 'text'
    )


def test_uri_deserializer():
    doc = Document(uri='https://jina.ai')

    assert (
        uri_getter(
            doc,
        )
        == 'https://jina.ai'
    )


def test_json_deserializer():
    doc = Document()
    doc._metadata[DocumentMetadata.JSON_TYPE] = ''
    doc.tags = 'attribute'
    assert json_getter(doc) == 'attribute'


def test_audio_deserializer():
    doc = Document()
    ad = Image.open(IMAGE_URI)
    doc.tensor = np.asarray(ad)

    assert np.all(audio_getter(doc) == Image.fromarray(np.asarray(ad)))
