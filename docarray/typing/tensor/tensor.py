from typing import Union, TypeVar, Any, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig

from docarray.document.base_node import BaseNode
from docarray.proto import DocumentProto, NdArrayProto, NodeProto
from pydantic import ValidationError

T = TypeVar('T', bound=np.ndarray)


class Tensor(np.ndarray, BaseNode):
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls: T, value: Union[T, Any], field: 'ModelField', config: 'BaseConfig') -> T:
        if isinstance(value, np.ndarray):
            return cls.from_ndarray(value)
        elif isinstance(value, Tensor):
            return value
        else:
            try:
                arr = np.ndarray(value)
                return cls.from_ndarray(arr)
            except Exception:
                pass  # handled below
        raise ValidationError(f'Expected a numpy.ndarray, got {type(value)}')

    @classmethod
    def from_ndarray(cls, value: np.ndarray) -> T:
        return value.view(cls)


    def _to_nested_item_protobuf(self) -> 'NodeProto':
        """Convert Document into a nested item protobuf message. This function should be called when the Document
        is nested into another Document that need to be converted into a protobuf

        :return: the nested item protobuf message
        """
        nd_proto = NdArrayProto()
        self.flush_ndarray(nd_proto, value=self)
        NodeProto(tensor=nd_proto)
        return NodeProto(tensor=nd_proto)

    @classmethod
    def read_ndarray(cls, pb_msg: 'NdArrayProto') -> 'Tensor':
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a numpy array
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(source.buffer, dtype=source.dtype)
            return cls.from_ndarray(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.from_ndarray(np.zeros(source.shape))
        else:
            raise ValueError(f'proto message {pb_msg} cannot be cast to a Tensor')

    @staticmethod
    def flush_ndarray(pb_msg: 'NdArrayProto', value: 'Tensor'):
        pb_msg.dense.buffer = value.tobytes()
        pb_msg.dense.ClearField('shape')
        pb_msg.dense.shape.extend(list(value.shape))
        pb_msg.dense.dtype = value.dtype.str