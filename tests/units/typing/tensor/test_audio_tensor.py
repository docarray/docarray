import os

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import Document
from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(1000, 2), AudioTorchTensor, torch.Tensor),
        (np.zeros((1000, 2)), AudioNdArray, np.ndarray),
    ],
)
def test_set_audio_tensor(tensor, cls_audio_tensor, cls_tensor):
    class MyAudioDoc(Document):
        tensor: cls_audio_tensor

    doc = MyAudioDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (AudioNdArray, np.zeros((1000, 2))),
        (AudioTorchTensor, torch.zeros(1000, 2)),
        (AudioTorchTensor, np.zeros((1000, 2))),
    ],
)
def test_validation(cls_tensor, tensor):
    arr = parse_obj_as(cls_tensor, tensor)
    assert isinstance(arr, cls_tensor)


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (AudioNdArray, torch.zeros(1000, 2)),
        (AudioNdArray, 'hello'),
        (AudioTorchTensor, 'hello'),
    ],
)
def test_illegal_validation(cls_tensor, tensor):
    match = str(cls_tensor).split('.')[-1][:-2]
    with pytest.raises(ValueError, match=match):
        parse_obj_as(cls_tensor, tensor)


@pytest.mark.parametrize(
    'cls_tensor,tensor,proto_key',
    [
        (AudioTorchTensor, torch.zeros(1000, 2), AudioTorchTensor._PROTO_FIELD_NAME),
        (AudioNdArray, np.zeros((1000, 2)), AudioNdArray._PROTO_FIELD_NAME),
    ],
)
def test_proto_tensor(cls_tensor, tensor, proto_key):
    tensor = parse_obj_as(cls_tensor, tensor)
    proto = tensor._to_node_protobuf()
    assert str(proto).startswith(proto_key)


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (AudioTorchTensor, torch.zeros(1000, 2)),
        (AudioNdArray, np.zeros((1000, 2))),
    ],
)
def test_save_audio_tensor_to_wav_file(cls_tensor, tensor, tmpdir):
    tmp_file = str(tmpdir / 'tmp.wav')
    audio_tensor = parse_obj_as(cls_tensor, tensor)
    audio_tensor.save_to_wav_file(tmp_file)
    assert os.path.isfile(tmp_file)
