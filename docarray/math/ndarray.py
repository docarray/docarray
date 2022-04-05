from typing import TYPE_CHECKING, Tuple, Sequence, Optional, List, Any

import numpy as np

if TYPE_CHECKING:
    from ..typing import ArrayType
    from .. import Document, DocumentArray


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


def ravel(value: 'ArrayType', docs: 'DocumentArray', field: str) -> None:
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
            docs[d.id, field] = row
    elif isinstance(value, (list, tuple)):
        for d, j in zip(docs, value):
            docs[d.id, field] = j
    else:

        emb_shape0 = value.shape[0]
        for d, j in zip(docs, range(emb_shape0)):
            docs[d.id, field] = value[j, ...]


def get_array_type(
    array: 'ArrayType', raise_error_if_not_array: bool = True
) -> Tuple[str, bool]:
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

    if raise_error_if_not_array:
        raise TypeError(f'can not determine the array type: {module_tags}.{class_name}')
    else:
        return 'python', False


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
    if framework == 'python':
        v = np.array(v)
    return v


def to_list(value) -> List[float]:
    r = to_numpy_array(value)
    if isinstance(r, np.ndarray):
        return r.tolist()
    elif isinstance(r, list):
        return r
    else:
        raise TypeError(f'{r} can not be converted into list')


def get_array_rows(array: 'ArrayType') -> Tuple[int, int]:
    """Get the number of rows of the ndarray without importing all frameworks

    :param array: input array
    :return: (num_rows, ndim)

    Examples

    >>> get_array_rows([1,2,3])
    1, 1
    >>> get_array_rows([[1,2,3], [4,5,6]])
    2, 2
    >>> get_array_rows([[1,2,3], [4,5,6], [7,8,9]])
    3, 2
    >>> get_array_rows(np.array([[1,2,3], [4,5,6], [7,8,9]]))
    3, 2
    """
    array_type, _ = get_array_type(array)

    if array_type == 'python':
        first_element_list_like = isinstance(array[0], (list, tuple))
        num_rows = len(array) if first_element_list_like else 1
        ndim = 2 if first_element_list_like else 1
    elif array_type in ('numpy', 'tensorflow', 'torch', 'paddle', 'scipy'):
        ndim = array.ndim
        if ndim == 1:
            num_rows = 1
        else:
            num_rows = array.shape[0]
    else:
        raise ValueError

    return num_rows, ndim


def check_arraylike_equality(x: 'ArrayType', y: 'ArrayType'):
    """Check if two array type objects are the same with the supported frameworks.

    Examples

    >>> import numpy as np
        x = np.array([[1,2,0,0,3],[1,2,0,0,3]])
        check_arraylike_equality(x,x)
    True

    >>> from scipy import sparse as sp
        x = sp.csr_matrix([[1,2,0,0,3],[1,2,0,0,3]])
        check_arraylike_equality(x,x)
    True

    >>> import torch
        x = torch.tensor([1,2,3])
        check_arraylike_equality(x,x)
    True
    """
    x_type, x_is_sparse = get_array_type(x)
    y_type, y_is_sparse = get_array_type(y)

    same_array = False
    if x_type == y_type and x_is_sparse == y_is_sparse:

        if x_type == 'python':
            same_array = x == y

        if x_type == 'numpy':
            # Numpy does not support sparse tensors
            import numpy as np

            same_array = np.array_equal(x, y)
        elif x_type == 'torch':
            import torch

            if x_is_sparse:
                # torch.equal NotImplementedError for sparse
                same_array = all((x - y).coalesce().values() == 0)
            else:
                same_array = torch.equal(x, y)
        elif x_type == 'scipy':
            # Not implemented in scipy this should work for all types
            # Note: you can't simply look at nonzero values because they can be in
            # different positions.
            if x.shape != y.shape:
                same_array = False
            else:
                same_array = (x != y).nnz == 0
        elif x_type == 'tensorflow':
            if x_is_sparse:
                same_array = x == y
            else:
                # Does not have equal implemented, only elementwise, therefore reduce .all is needed
                same_array = (x == y).numpy().all()
        elif x_type == 'paddle':
            # Paddle does not support sparse tensor on 11/8/2021
            # https://github.com/PaddlePaddle/Paddle/issues/36697
            # Does not have equal implemented, only elementwise, therefore reduce .all is needed
            same_array = (x == y).numpy().all()
        return same_array
    else:
        return same_array


def detach_tensor_if_present(x: Any) -> Any:
    """Check if input is a dense torch array and detaches the tensor from the current graph.
    :param array: input array
    :return: (num_rows, ndim)
    """
    x_type, x_sparse = get_array_type(x, raise_error_if_not_array=False)
    if x_type == 'torch' and x_sparse == False:
        import torch

        x = torch.tensor(x.detach().numpy())
    return x
