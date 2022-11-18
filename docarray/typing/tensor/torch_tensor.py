from typing import TYPE_CHECKING, Any, Dict, Type, TypeVar, Union, cast

import numpy as np
import torch  # type: ignore

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig
    import numpy as np

from docarray.document.base_node import BaseNode
from docarray.proto import NdArrayProto, NodeProto

T = TypeVar('T', bound='TorchTensor')

torch_base = type(torch.Tensor)  # type: Any
node_base = type(BaseNode)  # type: Any


class metaTorchAndNode(torch_base, node_base):
    pass


class TorchTensor(torch.Tensor, BaseNode, metaclass=metaTorchAndNode):
    # Subclassing torch.Tensor following the advice from here:
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, TorchTensor):
            return cast(T, value)
        elif isinstance(value, torch.Tensor):
            return cls.from_native_torch_tensor(value)

        else:
            try:
                arr: torch.Tensor = torch.tensor(value)
                return cls.from_native_torch_tensor(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a torch.Tensor, got {type(value)}')

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        # this is needed to dump to json
        field_schema.update(type='string', format='uuidhello')

    def _to_json_compatible(self):
        """
        Convert tensor into a json compatible object
        :return: a list representation of the tensor
        """
        return self.tolist()

    @classmethod
    def from_native_torch_tensor(cls: Type[T], value: torch.Tensor) -> T:
        """Create a TorchTensor from a native torch.Tensor

        :param value: the native torch.Tensor
        :return: a TorchTensor
        """
        value.__class__ = cls
        return cast(T, value)

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a TorchTensor from a numpy array

        :param value: the numpy array
        :return: a TorchTensor
        """
        return cls.from_native_torch_tensor(torch.from_numpy(value))

    def _to_node_protobuf(self: T, field: str = 'torch_tensor') -> NodeProto:
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
    def _read_from_proto(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
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
            raise ValueError(f'proto message {pb_msg} cannot be cast to a TorchTensor')

    @staticmethod
    def _flush_tensor_to_proto(pb_msg: 'NdArrayProto', value: 'TorchTensor'):
        value_np = value.detach().cpu().numpy()
        pb_msg.dense.buffer = value_np.tobytes()
        pb_msg.dense.ClearField('shape')
        pb_msg.dense.shape.extend(list(value_np.shape))
        pb_msg.dense.dtype = value_np.dtype.str
