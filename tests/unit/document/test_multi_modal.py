from docarray import Document
from docarray.types import TextDocument, ImageDocument
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
    ...


def test_with_tags():
    ...
