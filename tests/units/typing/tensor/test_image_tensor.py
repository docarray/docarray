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
import os

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.typing import ImageBytes, ImageNdArray, ImageTensor, ImageTorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.computation.tensorflow_backend import tnp
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
def test_torch_ndarray_to_image_tensor(tensor, cls_audio_tensor, cls_tensor):
    class MyImageDoc(BaseDoc):
        tensor: ImageTensor

    doc = MyImageDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_to_image_tensor():
    class MyImageDoc(BaseDoc):
        tensor: ImageTensor

    doc = MyImageDoc(tensor=tf.zeros((1000, 2)))
    assert isinstance(doc.tensor, ImageTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1000, 2)))
