# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from uuid import UUID

import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_doc.io.json import orjson_dumps
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


@pytest.mark.parametrize(
    'id', ['1234', 1234, UUID('cf57432e-809e-4353-adbd-9d5c0d733868')]
)
def test_operators(id):
    parsed_id = parse_obj_as(ID, id)
    assert parsed_id == str(id)
    assert parsed_id != 'aljdñjd'
    assert str(id)[0:1] in parsed_id
    assert 'docarray' not in parsed_id
