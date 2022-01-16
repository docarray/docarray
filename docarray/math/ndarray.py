from typing import TYPE_CHECKING, Tuple, Sequence, Optional, List

import numpy as np

if TYPE_CHECKING:
    from ..types import ArrayType
    from .. import Document


def unravel(docs: Sequence['Document'], field: str) -> Optional['ArrayType']:
    _first = getattr(docs[0], field)
    if _first is None:
        # failed to unravel, return as a list
        r = [getattr(d, field) for d in docs]
        if any(_rr is not None for _rr in r):
            return r
        else:
            return None

    framework, is_sparse = get_array_type(_first)
    all_fields = [getattr(d, field) for d in docs]
    cls_type = type(_first)

    if framework == 'python':
        return cls_type(all_fields)

    elif framework == 'numpy':
        return np.stack(all_fields)

    elif framework == 'tensorflow':
        import tensorflow as tf

        return tf.stack(all_fields)

    elif framework == 'torch':
        import torch

        return torch.stack(all_fields)

    elif framework == 'paddle':
        import paddle

        return paddle.stack(all_fields)

    elif framework == 'scipy':
        import scipy.sparse

        return cls_type(scipy.sparse.vstack(all_fields))


def ravel(value: 'ArrayType', docs: Sequence['Document'], field: str) -> None:
    """Ravel :attr:`value` into ``doc.field`` of each documents

    :param docs: the docs to set
    :param field: the field of the doc to set
    :param value: the value to be set on ``doc.field``
    """

    use_get_row = False
    if hasattr(value, 'getformat'):
        # for scipy only
        sp_format = value.getformat()
        if sp_format in {'bsr', 'coo'}:
            # for BSR and COO, they dont implement [j, ...] in scipy
            # but they offer get_row() API which implicitly translate the
            # sparse row into CSR format, hence needs to convert back
            # not very efficient, but this is the best we can do.
            use_get_row = True

    if use_get_row:
        emb_shape0 = value.shape[0]
        for d, j in zip(docs, range(emb_shape0)):
            row = getattr(value.getrow(j), f'to{sp_format}')()
            setattr(d, field, row)
    elif isinstance(value, (list, tuple)):
        for d, j in zip(docs, value):
            setattr(d, field, j)
    else:
        emb_shape0 = value.shape[0]
        for d, j in zip(docs, range(emb_shape0)):
            setattr(d, field, value[j, ...])


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

    if 'docarray' in module_tags:
        if class_name == 'NdArray':
            return 'docarray', False  # sparse or not is irrelevant

    if 'docarray_pb2' in module_tags:
        if class_name == 'NdArrayProto':
            return 'docarray_proto', False  # sparse or not is irrelevant

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


def to_numpy_array(value) -> 'np.ndarray':
    """Return the value always in :class:`numpy.ndarray` regardless the framework type.

    :return: the value in :class:`numpy.ndarray`.
    """
    v = value
    framework, is_sparse = get_array_type(value)
    if is_sparse:
        if hasattr(v, 'todense'):
            v = v.todense()
        elif hasattr(v, 'to_dense'):
            v = v.to_dense()
        elif framework == 'tensorflow':
            import tensorflow as tf

            if isinstance(v, tf.SparseTensor):
                v = tf.sparse.to_dense(v)

    if hasattr(v, 'numpy'):
        v = v.numpy()
    return v


def to_list(value) -> List[float]:
    r = to_numpy_array(value)
    if isinstance(r, np.ndarray):
        return r.tolist()
    elif isinstance(r, list):
        return r
    else:
        raise TypeError(f'{r} can not be converted into list')
