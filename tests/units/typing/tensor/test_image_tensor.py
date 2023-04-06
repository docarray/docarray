import os

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray.typing import ImageNdArray, ImageTorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing.tensor.image import ImageTensorFlowTensor


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (ImageTorchTensor, torch.zeros((224, 224, 3))),
        (ImageNdArray, np.zeros((224, 224, 3))),
    ],
)
def test_save_image_tensor_to_file(cls_tensor, tensor, tmpdir):
    tmp_file = str(tmpdir / 'tmp.jpg')
    image_tensor = parse_obj_as(cls_tensor, tensor)
    image_tensor.save(tmp_file)
    assert os.path.isfile(tmp_file)


@pytest.mark.tensorflow
def test_save_image_tensorflow_tensor_to_file(tmpdir):
    tmp_file = str(tmpdir / 'tmp.jpg')
    image_tensor = parse_obj_as(ImageTensorFlowTensor, tf.zeros((224, 224, 3)))
    image_tensor.save(tmp_file)
    assert os.path.isfile(tmp_file)
