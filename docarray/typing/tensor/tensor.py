from typing import TYPE_CHECKING, Any, Type, TypeVar, Union, cast

import numpy as np

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig

from docarray.document.base_node import BaseNode
from docarray.proto import NdArrayProto, NodeProto

T = TypeVar('T', bound='Tensor')


class Tensor(np.ndarray, BaseNode):
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T], value: Union[T, Any], field: 'ModelField', config: 'BaseConfig'
    ) -> T:
        if isinstance(value, np.ndarray):
            return cls.from_ndarray(value)
        elif isinstance(value, Tensor):
            return cast(T, value)
        else:
            try:
                arr: np.ndarray = np.ndarray(value)
                return cls.from_ndarray(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a numpy.ndarray, got {type(value)}')

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        return value.view(cls)

    def _to_node_protobuf(self: T, field: str = 'tensor') -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        nd_proto = NdArrayProto()
        self._flush_tensor_to_proto(nd_proto, value=self)
        return NodeProto(**{field: nd_proto})

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
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
    def _flush_tensor_to_proto(pb_msg: 'NdArrayProto', value: 'Tensor'):
        pb_msg.dense.buffer = value.tobytes()
        pb_msg.dense.ClearField('shape')
        pb_msg.dense.shape.extend(list(value.shape))
        pb_msg.dense.dtype = value.dtype.str
