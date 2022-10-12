from typing import Optional, Union

import pytest

from docarray import Document
from docarray.typing import Text, Image, Audio, JSON
from docarray.dataclasses.types import _is_optional, dataclass, _is_optional_field


def test_dataclass_opregetional_value():
    @dataclass
    class MultiModalDoc:
        foo: Text

    d = Document(MultiModalDoc(foo='sfdssf'))


def test_dataclass_optional_value():
    @dataclass
    class MultiModalDoc:
        foo: Optional[Text] = None

    d = Document(MultiModalDoc())
    assert d.bar.text is None
    d.bar.text = 'world'  # wont work

    assert len(d.chunks) == 1
    assert isinstance(d.bar, Document)


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
