from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...types import ArrayType


def flush_ndarray(pb_msg, value):
    framework, is_sparse = get_array_type(value)

    if framework == 'jina':
        # it is Jina's NdArray, simply copy it
        pb_msg.cls_name = 'numpy'
        pb_msg.CopyFrom(value._pb_body)
    elif framework == 'jina_proto':
        pb_msg.cls_name = 'numpy'
        pb_msg.CopyFrom(value)
    else:
        if is_sparse:
            if framework == 'scipy':
                pb_msg.parameters['sparse_format'] = value.getformat()
                _set_scipy_sparse(pb_msg, value)
            if framework == 'tensorflow':
                _set_tf_sparse(pb_msg, value)
            if framework == 'torch':
                _set_torch_sparse(pb_msg, value)
        else:
            if framework == 'numpy':
                pb_msg.cls_name = 'numpy'
                _set_dense_array(pb_msg.dense, value)
            if framework == 'python':
                pb_msg.cls_name = 'numpy'
                _set_dense_array(pb_msg.dense, np.array(value))
            if framework == 'tensorflow':
                pb_msg.cls_name = 'tensorflow'
                _set_dense_array(pb_msg.dense, value.numpy())
            if framework == 'torch':
                pb_msg.cls_name = 'torch'
                _set_dense_array(pb_msg.dense, value.detach().cpu().numpy())
            if framework == 'paddle':
                pb_msg.cls_name = 'paddle'
                _set_dense_array(pb_msg.dense, value.numpy())


def _set_dense_array(pb_msg, value):
    pb_msg.buffer = value.tobytes()
    pb_msg.ClearField('shape')
    pb_msg.shape.extend(list(value.shape))
    pb_msg.dtype = value.dtype.str


def _set_scipy_sparse(pb_msg, value):
    v = value.tocoo(copy=True)
    indices = np.stack([v.row, v.col], axis=1)
    _set_dense_array(pb_msg.sparse.indices, indices)
    _set_dense_array(pb_msg.sparse.values, v.data)
    pb_msg.sparse.ClearField('shape')
    pb_msg.sparse.shape.extend(v.shape)
    pb_msg.cls_name = 'scipy'


def _set_tf_sparse(pb_msg, value):
    _set_dense_array(pb_msg.sparse.indices, value.indices.numpy())
    _set_dense_array(pb_msg.sparse.values, value.values.numpy())
    pb_msg.sparse.ClearField('shape')
    pb_msg.sparse.shape.extend(value.shape)
    pb_msg.cls_name = 'tensorflow'


def _set_torch_sparse(pb_msg, value):
    _set_dense_array(pb_msg.sparse.indices, value.coalesce().indices().numpy())
    _set_dense_array(pb_msg.sparse.values, value.coalesce().values().numpy())
    pb_msg.sparse.ClearField('shape')
    pb_msg.sparse.shape.extend(list(value.size()))
    pb_msg.cls_name = 'torch'


def get_array_type(array: 'ArrayType') -> Tuple[str, bool]:
    """Get the type of ndarray without importing the framework

    :param array: any array, scipy, numpy, tf, torch, etc.
    :return: a tuple where the first element represents the framework, the second represents if it is sparse array
    """
    module_tags = array.__class__.__module__.split('.')
    class_name = array.__class__.__name__

    if isinstance(array, (list, tuple)):
        return 'python', False

    if 'numpy' in module_tags:
        return 'numpy', False

    if 'jina' in module_tags:
        if class_name == 'NdArray':
            return 'jina', False  # sparse or not is irrelevant

    if 'docarray_pb2' in module_tags:
        if class_name == 'NdArrayProto':
            return 'jina_proto', False  # sparse or not is irrelevant

    if 'tensorflow' in module_tags:
        if class_name == 'SparseTensor':
            return 'tensorflow', True
        if class_name == 'Tensor' or class_name == 'EagerTensor':
            return 'tensorflow', False

    if 'torch' in module_tags and class_name == 'Tensor':
        return 'torch', array.is_sparse

    if 'paddle' in module_tags and class_name == 'Tensor':
        # Paddle does not support sparse tensor on 11/8/2021
        # https://github.com/PaddlePaddle/Paddle/issues/36697
        return 'paddle', False

    if 'scipy' in module_tags and 'sparse' in module_tags:
        return 'scipy', True

    raise TypeError(f'can not determine the array type: {module_tags}.{class_name}')
