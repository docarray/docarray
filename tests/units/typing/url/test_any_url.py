from pydantic.tools import parse_obj_as

from docarray.typing import ImageUrl


def test_proto_any_url():

    uri = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')

    uri._to_node_protobuf()
