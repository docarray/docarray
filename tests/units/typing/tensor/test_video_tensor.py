import os
from io import BytesIO

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as

from docarray import BaseDocument
from docarray.typing import (
    AudioNdArray,
    AudioTorchTensor,
    VideoNdArray,
    VideoTorchTensor,
)
from docarray.utils.misc import is_tf_available

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
    class MyVideoDoc(BaseDocument):
        tensor: cls_video_tensor

    doc = MyVideoDoc(tensor=tensor)

    assert isinstance(doc.tensor, cls_video_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_set_video_tensor_tensorflow():
    class MyVideoDoc(BaseDocument):
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
    'cls_tensor,tensor',
    [
        (VideoNdArray, torch.zeros(1, 224, 224, 3)),
        (VideoTorchTensor, torch.zeros(224, 3)),
        (VideoTorchTensor, torch.zeros(1, 224, 224, 100)),
        (VideoNdArray, 'hello'),
        (VideoTorchTensor, 'hello'),
    ],
)
def test_illegal_validation(cls_tensor, tensor):
    match = str(cls_tensor).split('.')[-1][:-2]
    with pytest.raises(ValueError, match=match):
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
def test_save_video_tensor_to_bytes(video_tensor, tmpdir):
    b = BytesIO()
    video_tensor.save(b)
    isinstance(b, BytesIO)


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
