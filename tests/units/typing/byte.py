import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import Bytes


@pytest.mark.parametrize('value', [b'1234', '1234'])
def test_id_validation(value):
    parse_obj_as(Bytes, value)


def test_json_schema():
    schema_json_of(Bytes)


def test_dump_json():
    base_64 = parse_obj_as(Bytes, 1234)
    orjson_dumps(base_64)


def test_proto():
    base_64 = parse_obj_as(Bytes, 1234)
    base_64._to_node_protobuf()
