import json

from pydantic.tools import parse_obj_as, schema_json_of

from docarray.typing import AnyUrl


def test_proto_any_url():

    uri = parse_obj_as(AnyUrl, 'http://jina.ai/img.png')

    uri._to_node_protobuf()


def test_json_schema():
    schema_json_of(AnyUrl)


def test_dump_json(json_encoder):
    url = parse_obj_as(AnyUrl, 'http://jina.ai/img.png')
    json.dumps(url, cls=json_encoder)
