from typing import List

from docarray import Document
from docarray.document.mixins.multimodal import AttributeType
from docarray.types import TextDocument, ImageDocument, BlobDocument
from dataclasses import dataclass
import pytest
import numpy as np


def _assert_doc_schema(doc, schema):
    for field, attr_type, _type, position in schema:
        assert doc.metadata['multi_modal_schema'][field]['attribute_type'] == attr_type
        assert doc.metadata['multi_modal_schema'][field]['type'] == _type
        if position is not None:
            assert doc.metadata['multi_modal_schema'][field]['position'] == position
        else:
            assert 'position' not in doc.metadata['multi_modal_schema'][field]


def test_simple():
    @dataclass
    class MMDocument:
        title: TextDocument
        image: ImageDocument
        version: int

    obj = MMDocument(title='hello world', image=np.random.rand(10, 10, 3), version=20)
    doc = Document.from_dataclass(obj)
    assert doc.chunks[0].text == 'hello world'
    assert doc.chunks[1].tensor.shape == (10, 10, 3)
    assert doc.tags['version'] == 20

    assert 'multi_modal_schema' in doc.metadata

    expected_schema = [
        ('title', AttributeType.DOCUMENT, 'TextDocument', 0),
        ('image', AttributeType.DOCUMENT, 'ImageDocument', 1),
        ('version', AttributeType.PRIMITIVE, 'int', None),
    ]
    _assert_doc_schema(doc, expected_schema)


def test_nested():
    @dataclass
    class SubDocument:
        audio: BlobDocument
        date: str
        version: float

    @dataclass
    class MMDocument:
        sub_doc: SubDocument
        value: str
        image: ImageDocument

    obj = MMDocument(
        sub_doc=SubDocument(audio=b'1234', date='10.03.2022', version=1.5),
        image=np.random.rand(10, 10, 3),
        value='abc',
    )

    doc = Document.from_dataclass(obj)
    assert doc.tags['value'] == 'abc'

    assert doc.chunks[0].tags['date'] == '10.03.2022'
    assert doc.chunks[0].tags['version'] == 1.5
    assert doc.chunks[0].chunks[0].blob == b'1234'

    assert doc.chunks[1].tensor.shape == (10, 10, 3)

    assert 'multi_modal_schema' in doc.metadata

    expected_schema = [
        ('sub_doc', AttributeType.NESTED, 'SubDocument', 0),
        ('image', AttributeType.DOCUMENT, 'ImageDocument', 1),
        ('value', AttributeType.PRIMITIVE, 'str', None),
    ]
    _assert_doc_schema(doc, expected_schema)


def test_with_tags():
    @dataclass
    class MMDocument:
        attr1: str
        attr2: int
        attr3: float

    obj = MMDocument(attr1='123', attr2=10, attr3=1.1)

    doc = Document.from_dataclass(obj)
    assert doc.tags['attr1'] == '123'
    assert doc.tags['attr2'] == 10
    assert doc.tags['attr3'] == 1.1

    assert 'multi_modal_schema' in doc.metadata

    expected_schema = [
        ('attr1', AttributeType.PRIMITIVE, 'str', None),
        ('attr2', AttributeType.PRIMITIVE, 'int', None),
        ('attr3', AttributeType.PRIMITIVE, 'float', None),
    ]
    _assert_doc_schema(doc, expected_schema)


def test_iterable_doc():
    @dataclass
    class SocialPost:
        comments: List[str]
        ratings: List[int]
        images: List[ImageDocument]

    obj = SocialPost(
        comments=['hello world', 'goodbye world'],
        ratings=[1, 5, 4, 2],
        images=[np.random.rand(10, 10, 3) for _ in range(3)],
    )

    doc = Document.from_dataclass(obj)
    assert doc.tags['comments'] == ['hello world', 'goodbye world']
    assert doc.tags['ratings'] == [1, 5, 4, 2]
    assert len(doc.chunks[0].chunks) == 3
    for image_doc in doc.chunks[0].chunks:
        assert image_doc.tensor.shape == (10, 10, 3)

    assert 'multi_modal_schema' in doc.metadata

    expected_schema = [
        ('comments', AttributeType.ITERABLE_PRIMITIVE, 'List[str]', None),
        ('ratings', AttributeType.ITERABLE_PRIMITIVE, 'List[int]', None),
        ('images', AttributeType.ITERABLE_DOCUMENT, 'List[ImageDocument]', 0),
    ]
    _assert_doc_schema(doc, expected_schema)


def test_iterable_nested():
    @dataclass
    class SubtitleDocument:
        text: TextDocument
        frames: List[int]

    @dataclass
    class VideoDocument:
        frames: List[ImageDocument]
        subtitles: List[SubtitleDocument]

    obj = VideoDocument(
        frames=[np.random.rand(10, 10, 3) for _ in range(3)],
        subtitles=[
            SubtitleDocument(text='subtitle 0', frames=[0]),
            SubtitleDocument(text='subtitle 1', frames=[1, 2]),
        ],
    )

    doc = Document.from_dataclass(obj)

    assert len(doc.chunks) == 2
    assert len(doc.chunks[0].chunks) == 3
    for image_doc in doc.chunks[0].chunks:
        assert image_doc.tensor.shape == (10, 10, 3)

    assert len(doc.chunks[1].chunks) == 2
    for i, subtitle_doc in enumerate(doc.chunks[1].chunks):
        assert subtitle_doc.chunks[0].text == f'subtitle {i}'

    assert 'multi_modal_schema' in doc.metadata

    expected_schema = [
        ('text', AttributeType.DOCUMENT, 'TextDocument', 0),
        ('subtitles', AttributeType.ITERABLE_NESTED, 'List[SubtitleDocument]', 1),
    ]
    _assert_doc_schema(doc, expected_schema)

    expected_nested_schema = [
        ('frames', AttributeType.ITERABLE_DOCUMENT, 'List[ImageDocument]', 0),
        ('frames', AttributeType.ITERABLE_PRIMITIVE, 'List[int]', None),
    ]
    for subtitle in doc.chunks[1].chunks:
        _assert_doc_schema(doc.chunks[1], expected_schema)
