from typing import Optional, Union

import pytest

from docarray import Document
from docarray.typing import Image, Text, Audio, Video, Mesh, Tabular, Blob, JSON, URI
from docarray.dataclasses.types import _is_optional, dataclass


@pytest.mark.parametrize(
    'type_', [Image, Text, Audio, Video, Mesh, Tabular, Blob, JSON, URI]
)
def test_dataclass_optional_value(type_):
    @dataclass
    class MultiModalDoc:
        foo: Optional[type_] = None

    d = Document(MultiModalDoc())

    assert len(d.chunks) == 1
    assert isinstance(d.foo, Document)


def test_dataclass_optional_value_text():
    @dataclass
    class MultiModalDoc:
        foo: Optional[Text] = None

    d = Document(MultiModalDoc())
    assert (
        d.foo.text is Document(text=None).text
    )  # the None is automatically transform to '' in Document init
    d.foo.text = 'world'

    assert len(d.chunks) == 1
    assert isinstance(d.foo, Document)


def test_dataclass_optional_value_str():
    @dataclass
    class MultiModalDoc:
        foo: Optional[str] = None

    d = Document(MultiModalDoc())
    assert d.tags['foo'] is None
    d.tags['foo'] = 'world'

    assert len(d.chunks) == 0


def test_dataclass_optional_value_bytes():
    @dataclass
    class MultiModalDoc:
        foo: Optional[bytes] = None

    d = Document(MultiModalDoc())
    assert d.tags['foo'] is None
    assert len(d.chunks) == 0


@pytest.mark.parametrize('type_', [str, int, Text, Image, Audio, JSON])
def test_is_optional(type_):
    assert _is_optional(Optional[type])


@pytest.mark.parametrize('type_', [None, str, int, Text, Union[None, int, int]])
def test_is_not_optional(type_):
    assert not (_is_optional(type))
