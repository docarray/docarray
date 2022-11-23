from uuid import UUID

import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.document.io.json import orjson_dumps
from docarray.typing import ID


@pytest.mark.parametrize(
    'id', ['1234', 1234, UUID('cf57432e-809e-4353-adbd-9d5c0d733868')]
)
def test_id_validation(id):

    parsed_id = parse_obj_as(ID, id)

    assert parsed_id == str(id)


def test_json_schema():
    schema_json_of(ID)


def test_dump_json():
    id = parse_obj_as(ID, 1234)
    orjson_dumps(id)
