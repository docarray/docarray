import os

import numpy as np
import pytest
from PIL import Image
import io

from docarray.document.mixins.image import _to_image_tensor


@pytest.mark.parametrize(
    "width, height, output_shape",
    [
        (None, None, (50, 10, 3)),
        (8, None, (50, 8, 3)),
        (None, 30, (30, 10, 3)),
        (30, 8, (8, 30, 3)),
    ],
)
def test_to_image_tensor_pil(rgb_image_path, width, height, output_shape):
    tensor = _to_image_tensor(rgb_image_path, width, height)

    assert tensor.shape == output_shape
    assert isinstance(tensor, np.ndarray)


def test_to_image_tensor_blob(rgb_image_path):
    with open(rgb_image_path, "rb") as f:
        blob = io.BytesIO(f.read())

    tensor = _to_image_tensor(blob, None, None)

    assert tensor.shape == (50, 10, 3)
    assert isinstance(tensor, np.ndarray)


@pytest.fixture
def rgb_image_path(tmpdir):
    img_path = os.path.join(tmpdir, "image.png")
    RGB_COLOR_10X50_155_0_0 = Image.new("RGB", size=(10, 50), color=(0, 0, 0))
    RGB_COLOR_10X50_155_0_0.save(img_path)
    return img_path
