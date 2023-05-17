import os

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.computation.tensorflow_backend import tnp
from docarray.typing import ImageBytes, ImageNdArray, ImageTensor, ImageTorchTensor
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


@pytest.mark.parametrize(
    'image_tensor',
    [
        parse_obj_as(ImageTorchTensor, torch.zeros(224, 224, 3)),
        parse_obj_as(ImageNdArray, np.zeros((224, 224, 3))),
    ],
)
def test_save_image_tensor_to_bytes(image_tensor):
    b = image_tensor.to_bytes()
    isinstance(b, bytes)
    isinstance(b, ImageBytes)


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(1000, 2), ImageTorchTensor, torch.Tensor),
        (np.zeros((1000, 2)), ImageNdArray, np.ndarray),
    ],
)
def test_torch_ndarray_coercion(tensor, cls_audio_tensor, cls_tensor):
    class MyAudioDoc(BaseDoc):
        tensor: ImageTensor

    doc = MyAudioDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_coercion():
    class MyAudioDoc(BaseDoc):
        tensor: ImageTensor

    doc = MyAudioDoc(tensor=tf.zeros((1000, 2)))
    assert isinstance(doc.tensor, ImageTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1000, 2)))
