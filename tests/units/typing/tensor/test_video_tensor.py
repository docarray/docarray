import os

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as

from docarray import BaseDocument
from docarray.typing import VideoNdArray, VideoTorchTensor


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
            VideoTorchTensor._PROTO_FIELD_NAME,
        ),
        (VideoNdArray, np.zeros((1, 224, 224, 3)), VideoNdArray._PROTO_FIELD_NAME),
    ],
)
def test_proto_tensor(cls_tensor, tensor, proto_key):
    tensor = parse_obj_as(cls_tensor, tensor)
    proto = tensor._to_node_protobuf()
    assert str(proto).startswith(proto_key)


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (VideoTorchTensor, torch.zeros(1, 224, 224, 3)),
        (VideoNdArray, np.zeros((1, 224, 224, 3))),
    ],
)
def test_save_video_tensor_to_file(cls_tensor, tensor, tmpdir):
    tmp_file = str(tmpdir / 'tmp.mp4')
    video_tensor = parse_obj_as(cls_tensor, tensor)
    video_tensor.save_to_mp4_file(tmp_file)
    assert os.path.isfile(tmp_file)
