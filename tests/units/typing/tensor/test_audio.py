import os

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor
from tests import TOYDATA_DIR


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (AudioTorchTensor, torch.zeros(3, 224, 224)),
        (AudioNdArray, np.zeros((3, 224, 224))),
    ],
)
def test_proto_tensor(cls_tensor, tensor):

    tensor = parse_obj_as(cls_tensor, tensor)

    tensor._to_node_protobuf()


@pytest.mark.parametrize(
    'cls_tensor,tensor',
    [
        (AudioTorchTensor, torch.zeros(3, 224, 224)),
        (AudioNdArray, np.zeros((3, 224, 224))),
    ],
)
def test_save_audio_tensor_to_file(cls_tensor, tensor):
    tmp_file = str(TOYDATA_DIR / 'tmp.wav')
    audio_tensor = parse_obj_as(cls_tensor, tensor)
    audio_tensor.save_audio_tensor_to_file(tmp_file)
    assert os.path.isfile(tmp_file)
    os.remove(tmp_file)
