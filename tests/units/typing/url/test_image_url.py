import numpy as np
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import ImageUrl


def test_image_url():
    uri = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')

    tensor = uri.load()

    assert isinstance(tensor, np.ndarray)


def test_proto_image_url():

    uri = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')

    uri._to_node_protobuf()


def test_json_schema():
    schema_json_of(ImageUrl)


def test_dump_json():
    url = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')
    orjson_dumps(url)
