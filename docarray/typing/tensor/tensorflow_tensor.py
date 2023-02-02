from typing import TYPE_CHECKING, Any, Dict, Generic, Type, TypeVar, Union, cast

import numpy as np
import tensorflow as tf  # type: ignore

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


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaTensorFlow(
    AbstractTensor.__parametrized_meta__,  # type: ignore
    node_base,  # type: ignore
    tf_base,  # type: ignore
):  # type: ignore
    pass


@_register_proto(proto_type_name='tensorflow_tensor')
class TensorFlowTensor(AbstractTensor, Generic[ShapeT], metaclass=metaTensorFlow):
    """
    TensorFlowTensor class with a `.tensor` attribute of type `tf.Tensor`, intended for
    use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coersion from compatible types like numpy.ndarray.

    This type can also be used in a parametrized way, specifying the shape of the
    tensor.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.typing import TensorFlowTensor
        import tensorflow as tf


        class MyDoc(BaseDocument):
            tensor: TensorFlowTensor
            image_tensor: TensorFlowTensor[3, 224, 224]
            square_crop: TensorFlowTensor[3, 'x', 'x']


        # create a document with tensors
        doc = MyDoc(
            tensor=tf.zeros((128,)),
            image_tensor=tf.zeros((3, 224, 224)),
            square_crop=tf.zeros((3, 64, 64)),
        )

        # automatic shape conversion
        doc = MyDoc(
            tensor=tf.zeros((128,)),
            image_tensor=tf.zeros((224, 224, 3)),  # will reshape to (3, 224, 224)
            square_crop=tf.zeros((3, 128, 128)),
        )

        # !! The following will raise an error due to shape mismatch !!
        doc = MyDoc(
            tensor=tf.zeros((128,)),
            image_tensor=tf.zeros((224, 224)),  # this will fail validation
            square_crop=tf.zeros((3, 128, 64)),  # this will also fail validation
        )

    If you want to call functions provided by tensorflow you have to access the
    `.tensor` attribute or call `.unwrap()` on your TensorFlowTensor instance:

    .. code-block:: python
        from docarray.typing import TensorFlowTensor
        import tensorflow as tf


        t = TensorFlowTensor(tf.zeros((224, 224)))

        # tensorflow functions
        broadcasted = tf.broadcast_to(t.tensor, (3, 224, 224))
        broadcasted = tf.broadcast_to(t.unwrap(), (3, 224, 224))
        broadcasted = tf.broadcast_to(t, (3, 224, 224))  # this will fail

        # tensorflow.Tensor methods:
        arr = t.tensor.numpy()
        arr = t.unwrap().numpy()
        arr = t.numpy()  # this will fail

    """

    __parametrized_meta__ = metaTensorFlow

    def __init__(self, tensor: tf.Tensor):
        super().__init__()
        self.tensor = tensor

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
        if isinstance(value, TensorFlowTensor):
            return cast(T, value)
        elif isinstance(value, tf.Tensor):
            return cls._docarray_from_native(value)
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
    def _docarray_from_native(cls: Type[T], value: Union[tf.Tensor, T]) -> T:
        """
        Create a TensorFlowTensor from a tensorflow.Tensor or TensorFlowTensor
        instance.

        :param value: instance of tf.Tensor or TensorFlowTensor
        :return: a TensorFlowTensor
        """
        if isinstance(value, TensorFlowTensor):
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
                value.__class__ = cls.__unparametrizedcls__
            else:
                value.__class__ = cls
            return cast(T, value)
        else:
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
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
        """
        Transform self into an NdArrayProto protobuf message.
        """
        raise NotImplementedError

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        Read ndarray from a proto msg.
        :param pb_msg:
        :return: a TensorFlowTensor
        """

        raise NotImplementedError

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a TensorFlowTensor from a numpy array.

        :param value: the numpy array
        :return: a TensorFlowTensor
        """
        return cls._docarray_from_native(tf.convert_to_tensor(value))

    def unwrap(self) -> tf.Tensor:
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


        :return: a tf.Tensor
        """
        return self.tensor
