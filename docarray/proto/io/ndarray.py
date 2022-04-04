from typing import TYPE_CHECKING, Optional

import numpy as np

from ...math.ndarray import get_array_type, to_numpy_array

if TYPE_CHECKING:
    from ...typing import ArrayType
    from ..docarray_pb2 import NdArrayProto


def read_ndarray(pb_msg: 'NdArrayProto') -> 'ArrayType':
    is_sparse = pb_msg.WhichOneof('content') == 'sparse'
    framework = pb_msg.cls_name

    if is_sparse:
        if framework == 'scipy':
            idx, val, shape = _get_raw_sparse_array(pb_msg)
            from scipy.sparse import coo_matrix

            x = coo_matrix((val, idx.T), shape=shape)
            sp_format = pb_msg.parameters['sparse_format']
            if sp_format == 'bsr':
                return x.tobsr()
            elif sp_format == 'csc':
                return x.tocsc()
            elif sp_format == 'csr':
                return x.tocsr()
            elif sp_format == 'coo':
                return x
        elif framework == 'tensorflow':
            idx, val, shape = _get_raw_sparse_array(pb_msg)
            from tensorflow import SparseTensor

            return SparseTensor(idx, val, shape)
        elif framework == 'torch':
            idx, val, shape = _get_raw_sparse_array(pb_msg)
            from torch import sparse_coo_tensor

            return sparse_coo_tensor(idx, val, shape)
    else:
        if framework in {'numpy', 'torch', 'paddle', 'tensorflow', 'list'}:
            x = _get_dense_array(pb_msg.dense)
            return _to_framework_array(x, framework)


def flush_ndarray(
    pb_msg: 'NdArrayProto', value: 'ArrayType', ndarray_type: Optional[str] = None
):
    if ndarray_type == 'list':
        value = to_numpy_array(value).tolist()
    elif ndarray_type == 'numpy':
        value = to_numpy_array(value)

    framework, is_sparse = get_array_type(value)

    if framework == 'docarray':
        # it is Jina's NdArray, simply copy it
        pb_msg.cls_name = 'numpy'
        pb_msg.CopyFrom(value)
    elif framework == 'docarray_proto':
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
                pb_msg.cls_name = 'list'
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


def _get_raw_sparse_array(pb_msg):
    idx = _get_dense_array(pb_msg.sparse.indices)
    val = _get_dense_array(pb_msg.sparse.values)
    shape = list(pb_msg.sparse.shape)
    return idx, val, shape


def _get_dense_array(source):
    if source.buffer:
        x = np.frombuffer(source.buffer, dtype=source.dtype)
        return x.reshape(source.shape)
    elif len(source.shape) > 0:
        return np.zeros(source.shape)


def _to_framework_array(x, framework):
    if framework == 'numpy':
        return x
    elif framework == 'tensorflow':
        from tensorflow import convert_to_tensor

        return convert_to_tensor(x)
    elif framework == 'torch':
        from torch import from_numpy

        return from_numpy(x)
    elif framework == 'paddle':
        from paddle import to_tensor

        return to_tensor(x)
    elif framework == 'list':
        return x.tolist()
