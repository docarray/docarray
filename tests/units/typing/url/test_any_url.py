import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import AnyUrl


@pytest.mark.proto
def test_proto_any_url():
    uri = parse_obj_as(AnyUrl, 'http://jina.ai/img.png')

    uri._to_node_protobuf()


def test_json_schema():
    schema_json_of(AnyUrl)


def test_dump_json():
    url = parse_obj_as(AnyUrl, 'http://jina.ai/img.png')
    orjson_dumps(url)


@pytest.mark.parametrize(
    'relative_path',
    [
        'data/05978.jpg',
        '../../data/05978.jpg',
    ],
)
def test_relative_path(relative_path):
    # see issue: https://github.com/docarray/docarray/issues/978
    url = parse_obj_as(AnyUrl, relative_path)
    assert url == relative_path


def test_operators():
    url = parse_obj_as(AnyUrl, 'data/05978.jpg')
    assert url == 'data/05978.jpg'
    assert url != 'aljdñjd'
    assert 'data' in url
    assert 'docarray' not in url
