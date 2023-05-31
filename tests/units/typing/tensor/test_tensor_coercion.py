import numpy as np
import pytest
import tensorflow as tf
import torch
from pydantic import parse_obj_as

from docarray.typing import NdArray, TensorFlowTensor, TorchTensor

pure_tensor_to_test = [
    np.zeros((3, 224, 224)),
    torch.zeros(3, 224, 224),
    tf.zeros((3, 224, 224)),
]
docarray_tensor_to_test = [
    NdArray._docarray_from_native(np.zeros((3, 224, 224))),
    TorchTensor._docarray_from_native(torch.zeros(3, 224, 224)),
    TensorFlowTensor._docarray_from_native(tf.zeros((3, 224, 224))),
]


@pytest.mark.tensorflow
@pytest.mark.parametrize('tensor', pure_tensor_to_test + docarray_tensor_to_test)
@pytest.mark.parametrize('tensor_cls', [NdArray, TorchTensor, TensorFlowTensor])
def test_torch_tensor_coerse(tensor_cls, tensor):
    t = parse_obj_as(tensor_cls, tensor)
    assert isinstance(t, tensor_cls)
