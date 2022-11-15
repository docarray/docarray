from uuid import UUID

import pytest
from pydantic.tools import parse_obj_as

from docarray.typing import ID


@pytest.mark.parametrize(
    'id', ['1234', 1234, UUID('cf57432e-809e-4353-adbd-9d5c0d733868')]
)
def test_id_validation(id):

    parsed_id = parse_obj_as(ID, id)

    assert parsed_id == str(id)
