from typing import TYPE_CHECKING, Any, Dict, Generic, Type, TypeVar, Union

import numpy as np
import tensorflow as tf

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig
    from docarray.proto import NdArrayProto
    from docarray.computation.tensorflow_backend import TensorFlowCompBackend

from docarray.base_document.base_node import BaseNode

T = TypeVar('T', bound='TensorFlowTensor')
ShapeT = TypeVar('ShapeT')

tf_base: type = type(tf.Tensor)
node_base: type = type(BaseNode)


class metaTensorFlow(
    AbstractTensor.__parametrized_meta__,  # type: ignore
    node_base,  # type: ignore
    tf_base,  # type: ignore
):  # type: ignore
    pass


@_register_proto(proto_type_name='tensorflow_tensor')
class TensorFlowTensor(AbstractTensor, Generic[ShapeT], metaclass=metaTensorFlow):

    __parametrized_meta__ = metaTensorFlow

    def __init__(self, tensor: tf.Tensor):
        super().__init__()
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

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
        if isinstance(value, tf.Tensor):
            return cls(tensor=value)
        else:
            try:
                arr: tf.Tensor = tf.constant(value)
                return cls(tensor=arr)
            except Exception:
                pass  # handled below
        raise ValueError(
            f'Expected a tensorflow.Tensor compatible type, got {type(value)}'
        )

    @classmethod
    def _docarray_from_native(cls: Type[T], value: tf.Tensor) -> T:
        """Create a TensorFlowTensor from a native tensorflow.Tensor

        :param value: the native tf.Tensor
        :return: a TensorFlowTensor
        """
        if cls.__unparametrizedcls__:  # This is not None if the tensor is parametrized
            cls_param = cls.__unparametrizedcls__
        else:
            cls_param = cls
        return cls_param(tensor=value)

    @staticmethod
    def get_comp_backend() -> 'TensorFlowCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.tensorflow_backend import TensorFlowCompBackend

        return TensorFlowCompBackend()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        # this is needed to dump to json
        field_schema.update(type='string', format='tensor')

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert TensorFlowTensor into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.unwrap().numpy()

    def to_protobuf(self) -> 'NdArrayProto':
        pass

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        pass

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a TensorFlowTensor from a numpy array.

        :param value: the numpy array
        :return: a TensorFlowTensor
        """
        return cls._docarray_from_native(tf.convert_to_tensor(value))

    def unwrap(self):
        """
        Return the original tensorflow.Tensor without any memory copy.

        The original view rest intact and is still a Document TensorFlowTensor
        but the return object is a pure tf.Tensor but both object share
        the same memory layout.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray.typing import TensorFlowTensor
            import tensorflow as tf

            t1 = TensorFlowTensor.validate(tf.zeros((3, 224, 224)), None, None)
            # here t1 is a docarray TensorFlowTensor
            t2 = t.unwrap()
            # here t2 is a pure tf.Tensor but t1 is still a Docarray TensorFlowTensor
            # But both share the same underlying memory


        :return: a tf.Tensor
        """
        return self.tensor
