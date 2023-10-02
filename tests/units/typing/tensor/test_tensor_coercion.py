import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray.typing import NdArray, TorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing import TensorFlowTensor
else:

    ### This is needed to fake the import of tensorflow when it is not installed
    class TfNotInstalled:
        def zeros(self, *args, **kwargs):
            return 0

    class TensorFlowTensor:
        def _docarray_from_native(self, *args, **kwargs):
            return 0

    tf = TfNotInstalled()


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

    t_numpy = t._docarray_to_ndarray()
    assert t_numpy.shape == (3, 224, 224)
    assert (t_numpy == np.zeros((3, 224, 224))).all()
