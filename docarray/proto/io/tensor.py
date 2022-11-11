import numpy as np

from docarray.proto import NdArrayProto
from docarray.typing.ndarray import Tensor


def read_ndarray(pb_msg: 'NdArrayProto') -> 'Tensor':
    """
    read ndarray from a proto msg
    :param pb_msg:
    :return: a numpy array
    """
    source = pb_msg.dense
    if source.buffer:
        x = np.frombuffer(source.buffer, dtype=source.dtype)
        return x.reshape(source.shape)
    elif len(source.shape) > 0:
        return np.zeros(source.shape)
    else:
        raise ValueError(f'proto message {pb_msg} cannot be cast to a Tensor')


def flush_ndarray(pb_msg: 'NdArrayProto', value: 'Tensor'):
    pb_msg.dense.buffer = value.tobytes()
    pb_msg.dense.ClearField('shape')
    pb_msg.dense.shape.extend(list(value.shape))
    pb_msg.dense.dtype = value.dtype.str
