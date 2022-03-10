from typing import List

from docarray import Document
from docarray.types import TextDocument, ImageDocument, BlobDocument
from dataclasses import dataclass
import pytest
import numpy as np


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


def test_nested():
    @dataclass
    class SubDocument:
        audio: BlobDocument
        date: str
        version: float

    @dataclass
    class MMDocument:
        sub_doc: SubDocument
        image: ImageDocument
        value: str

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


def test_iterable():
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
