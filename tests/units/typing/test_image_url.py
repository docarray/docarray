from pydantic.tools import parse_obj_as

from docarray.typing import ImageUrl, Tensor


def test_image_url():
    uri = parse_obj_as(ImageUrl, 'http://jina.ai/img.png')

    tensor = uri.load()

    assert isinstance(tensor, Tensor)
