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
from pydantic.tools import parse_obj_as

from docarray import BaseDoc
from docarray.typing import (
    AudioNdArray,
    AudioTorchTensor,
    VideoBytes,
    VideoNdArray,
    VideoTensor,
    VideoTorchTensor,
)
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

    from docarray.typing.tensor.video import VideoTensorFlowTensor


@pytest.mark.parametrize(
    'tensor,cls_video_tensor,cls_tensor',
    [
        (torch.zeros(1, 224, 224, 3), VideoTorchTensor, torch.Tensor),
        (np.zeros((1, 224, 224, 3)), VideoNdArray, np.ndarray),
    ],
)
def test_set_video_tensor(tensor, cls_video_tensor, cls_tensor):
    class MyVideoDoc(BaseDoc):
        tensor: cls_video_tensor

    doc = MyVideoDoc(tensor=tensor)

    assert isinstance(doc.tensor, cls_video_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_set_video_tensor_tensorflow():
    class MyVideoDoc(BaseDoc):
        tensor: VideoTensorFlowTensor

    doc = MyVideoDoc(tensor=tf.zeros((1, 224, 224, 3)))

    assert isinstance(doc.tensor, VideoTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1, 224, 224, 3)))


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (VideoNdArray, np.zeros((1, 224, 224, 3))),
        (VideoTorchTensor, torch.zeros(1, 224, 224, 3)),
        (VideoTorchTensor, np.zeros((1, 224, 224, 3))),
    ],
)
def test_validation(cls_tensor, tensor):
    arr = parse_obj_as(cls_tensor, tensor)
    assert isinstance(arr, cls_tensor)


@pytest.mark.tensorflow
def test_validation_tensorflow():
    arr = parse_obj_as(VideoTensorFlowTensor, np.zeros((1, 224, 224, 3)))
    assert isinstance(arr, VideoTensorFlowTensor)

    arr = parse_obj_as(VideoTensorFlowTensor, tf.zeros((1, 224, 224, 3)))
    assert isinstance(arr, VideoTensorFlowTensor)

    arr = parse_obj_as(VideoTensorFlowTensor, torch.zeros((1, 224, 224, 3)))
    assert isinstance(arr, VideoTensorFlowTensor)


@pytest.mark.parametrize(
    'cls_tensor,tensor,expect_error',
    [
        (VideoNdArray, torch.zeros(1, 224, 224, 3), False),
        (VideoNdArray, torch.zeros(1, 224, 224, 100), True),
        (VideoTorchTensor, torch.zeros(1, 224, 224, 3), False),
        (VideoTorchTensor, torch.zeros(1, 224, 224, 100), True),
        (VideoNdArray, 'hello', True),
        (VideoTorchTensor, 'hello', True),
    ],
)
def test_illegal_validation(cls_tensor, tensor, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            parse_obj_as(cls_tensor, tensor)
    else:
        parse_obj_as(cls_tensor, tensor)


@pytest.mark.parametrize(
    'cls_tensor,tensor,proto_key',
    [
        (
            VideoTorchTensor,
            torch.zeros(1, 224, 224, 3),
            VideoTorchTensor._proto_type_name,
        ),
        (VideoNdArray, np.zeros((1, 224, 224, 3)), VideoNdArray._proto_type_name),
    ],
)
def test_proto_tensor(cls_tensor, tensor, proto_key):
    tensor = parse_obj_as(cls_tensor, tensor)
    proto = tensor._to_node_protobuf()
    assert proto_key in str(proto)


@pytest.mark.tensorflow
def test_proto_tensor_tensorflow():
    tensor = parse_obj_as(VideoTensorFlowTensor, tf.zeros((1, 224, 224, 3)))
    proto = tensor._to_node_protobuf()
    assert VideoTensorFlowTensor._proto_type_name in str(proto)


@pytest.mark.parametrize(
    'video_tensor',
    [
        parse_obj_as(VideoTorchTensor, torch.zeros(1, 224, 224, 3)),
        parse_obj_as(VideoNdArray, np.zeros((1, 224, 224, 3))),
    ],
)
def test_save_video_tensor_to_file(video_tensor, tmpdir):
    tmp_file = str(tmpdir / 'tmp.mp4')
    video_tensor.save(tmp_file)
    assert os.path.isfile(tmp_file)


@pytest.mark.parametrize(
    'video_tensor',
    [
        parse_obj_as(VideoTorchTensor, torch.zeros(1, 224, 224, 3)),
        parse_obj_as(VideoNdArray, np.zeros((1, 224, 224, 3))),
    ],
)
def test_save_video_tensor_to_bytes(video_tensor):
    b = video_tensor.to_bytes()
    isinstance(b, bytes)
    isinstance(b, VideoBytes)


@pytest.mark.tensorflow
def test_save_video_tensorflow_tensor_to_file(tmpdir):
    tmp_file = str(tmpdir / 'tmp.mp4')
    video_tensor = parse_obj_as(VideoTensorFlowTensor, tf.zeros((1, 224, 224, 3)))
    video_tensor.save(tmp_file)
    assert os.path.isfile(tmp_file)


@pytest.mark.parametrize(
    'video_tensor',
    [
        parse_obj_as(VideoTorchTensor, torch.zeros(1, 224, 224, 3)),
        parse_obj_as(VideoNdArray, np.zeros((1, 224, 224, 3))),
    ],
)
@pytest.mark.parametrize(
    'audio_tensor',
    [
        parse_obj_as(AudioTorchTensor, torch.randn(100, 1, 1024).to(torch.float32)),
        parse_obj_as(AudioNdArray, np.random.randn(100, 1, 1024).astype('float32')),
    ],
)
def test_save_video_tensor_to_file_including_audio(video_tensor, audio_tensor, tmpdir):
    tmp_file = str(tmpdir / 'tmp.mp4')
    video_tensor.save(tmp_file, audio_tensor=audio_tensor)
    assert os.path.isfile(tmp_file)


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(2, 10, 10, 3), VideoTorchTensor, torch.Tensor),
        (np.zeros((2, 10, 10, 3)), VideoNdArray, np.ndarray),
    ],
)
def test_torch_ndarray_to_video_tensor(tensor, cls_audio_tensor, cls_tensor):
    class MyAudioDoc(BaseDoc):
        tensor: VideoTensor

    doc = MyAudioDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_to_video_tensor():
    class MyAudioDoc(BaseDoc):
        tensor: VideoTensor

    doc = MyAudioDoc(tensor=tf.zeros((2, 10, 10, 3)))
    assert isinstance(doc.tensor, VideoTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((2, 10, 10, 3)))
