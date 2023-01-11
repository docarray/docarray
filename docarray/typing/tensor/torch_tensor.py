import warnings
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, Generic, Tuple, Type, TypeVar, Union, cast

import numpy as np
import torch  # type: ignore

from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig
    import numpy as np
    from docarray.proto import NdArrayProto, NodeProto
    from docarray.computation.torch_backend import TorchCompBackend

from docarray.base_document.base_node import BaseNode

T = TypeVar('T', bound='TorchTensor')
ShapeT = TypeVar('ShapeT')

torch_base = type(torch.Tensor)  # type: Any
node_base = type(BaseNode)  # type: Any


class metaTorchAndNode(torch_base, node_base):
    pass


class TorchTensor(
    torch.Tensor, AbstractTensor, Generic[ShapeT], metaclass=metaTorchAndNode
):
    # Subclassing torch.Tensor following the advice from here:
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    """
    Subclass of torch.Tensor, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coersion from compatible types like numpy.ndarray.

    This type can also be used in a parametrized way,
    specifying the shape of the tensor.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.typing import TorchTensor
        import torch


        class MyDoc(BaseDocument):
            tensor: TorchTensor
            image_tensor: TorchTensor[3, 224, 224]


        # create a document with tensors
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(3, 224, 224),
        )

        # automatic shape conversion
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(224, 224, 3),  # will reshape to (3, 224, 224)
        )

        # !! The following will raise an error due to shape mismatch !!
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(224, 224),  # this will fail validation
        )

    """

    __parametrized_meta__ = metaTorchAndNode
    _PROTO_FIELD_NAME = 'torch_tensor'

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __docarray_validate_shape__(cls, t: T, shape: Tuple[int]) -> T:  # type: ignore
        if t.shape == shape:
            return t
        elif any(isinstance(dim, str) for dim in shape):
            known_dims: Dict[str, int] = {}
            for tdim, dim in zip(t.shape, shape):
                if isinstance(dim, int) and tdim != dim:
                    raise ValueError(f"Tensor shape mismatch. Expected {shape}, got {t.shape}")
                elif isinstance(dim, str):
                    if dim in known_dims and known_dims[dim] != tdim:
                        raise ValueError(f"Tensor shape mismatch. Expected {shape}, got {t.shape}")
                    else:
                        known_dims[dim] = tdim
            else:
                return t
        else:
            warnings.warn(
                f'Tensor shape mismatch. Reshaping tensor '
                f'of shape {t.shape} to shape {shape}'
            )
            try:
                value = cls.__docarray_from_native__(t.view(shape))
                return cast(T, value)
            except RuntimeError:
                raise ValueError(
                    f'Cannot reshape tensor of shape {t.shape} to shape {shape}'
                )

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
            return cls.__docarray_from_native__(value)

        else:
            try:
                arr: torch.Tensor = torch.tensor(value)
                return cls.__docarray_from_native__(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a torch.Tensor compatible type, got {type(value)}')

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        # this is needed to dump to json
        field_schema.update(type='string', format='tensor')

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert torchTensor into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.numpy()  ## might need to  check device later

    def unwrap(self) -> torch.Tensor:
        """
        Return the original torch.Tensor without any memory copy.

        The original view rest intact and is still a Document TorchTensor
        but the return object is a pure torch Tensor but both object share
        the same memory layout.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray.typing import TorchTensor
            import torch

            t = TorchTensor.validate(torch.zeros(3, 224, 224), None, None)
            # here t is a docarray TorchTensor
            t2 = t.unwrap()
            # here t2 is a pure torch.Tensor but t1 is still a Docarray TorchTensor
            # But both share the same underlying memory


        :return: a torch Tensor
        """
        value = copy(self)  # as unintuitive as it sounds, this
        # does not do any relevant memory copying, just shallow
        # reference to the torch data
        value.__class__ = torch.Tensor  # type: ignore
        return value

    @classmethod
    def __docarray_from_native__(cls: Type[T], value: torch.Tensor) -> T:
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
        return cls.__docarray_from_native__(torch.from_numpy(value))

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(**{self._PROTO_FIELD_NAME: nd_proto})

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a torch tensor
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls.from_ndarray(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.from_ndarray(np.zeros(source.shape))
        else:
            raise ValueError(f'proto message {pb_msg} cannot be cast to a TorchTensor')

    def to_protobuf(self) -> 'NdArrayProto':
        """
        transform self into a NdArrayProto protobuf message
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        value_np = self.detach().cpu().numpy()
        nd_proto.dense.buffer = value_np.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(value_np.shape))
        nd_proto.dense.dtype = value_np.dtype.str

        return nd_proto

    @staticmethod
    def get_comp_backend() -> Type['TorchCompBackend']:
        """Return the computational backend of the tensor"""
        from docarray.computation.torch_backend import TorchCompBackend

        return TorchCompBackend
