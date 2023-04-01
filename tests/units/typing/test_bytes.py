import os

from pydantic import parse_obj_as

from docarray.typing import ImageBytes, ImageTensor, ImageUrl

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_IMAGE_DATA = os.path.join(CUR_DIR, '..', '..', 'toydata', 'image-data')
IMAGE_PATHS = {
    'png': os.path.join(PATH_TO_IMAGE_DATA, 'so_good.png'),
    'jpg': os.path.join(PATH_TO_IMAGE_DATA, '05984.jpg'),
    'jpeg': os.path.join(PATH_TO_IMAGE_DATA, '05984-2.jpeg'),
}


def test_bytes_load():
    url = parse_obj_as(ImageUrl, IMAGE_PATHS['png'])

    tensor = parse_obj_as(ImageTensor, url.load())

    bytes_ = parse_obj_as(ImageBytes, tensor.to_bytes())

    tensor_new = parse_obj_as(ImageTensor, bytes_.load())

    assert (tensor_new == tensor).all()
