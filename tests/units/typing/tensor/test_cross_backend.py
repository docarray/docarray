import numpy as np
from pydantic import parse_obj_as

from docarray.typing import NdArray, TensorFlowTensor, TorchTensor


def test_coercion_behavior():
    t_np = parse_obj_as(NdArray[128], np.zeros(128))
    t_th = parse_obj_as(TorchTensor[128], np.zeros(128))
    t_tf = parse_obj_as(TensorFlowTensor[128], np.zeros(128))

    assert isinstance(t_np, NdArray[128])
    assert not isinstance(t_np, TensorFlowTensor[128])
    assert not isinstance(t_np, TorchTensor[128])

    assert isinstance(t_th, TorchTensor[128])
    assert not isinstance(t_th, NdArray[128])
    assert not isinstance(t_th, TensorFlowTensor[128])

    assert isinstance(t_tf, TensorFlowTensor[128])
    assert not isinstance(t_tf, TorchTensor[128])
    assert not isinstance(t_tf, NdArray[128])
