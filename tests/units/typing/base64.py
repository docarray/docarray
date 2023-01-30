import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import Base64


@pytest.mark.parametrize('value', [b'1234', '1234'])
def test_validation(value):
    parse_obj_as(Base64, value)


def test_decode():
    value = b'1234'
    base_64 = parse_obj_as(Base64, value)
    assert value == base_64.decode()
    assert value.decode() == base_64.decode_str()


def test_json_schema():
    schema_json_of(Base64)


def test_dump_json():
    bytes_ = parse_obj_as(Base64, 1234)
    orjson_dumps(bytes_)


def test_proto():
    bytes_ = parse_obj_as(Base64, 1234)
    bytes_._to_node_protobuf()
