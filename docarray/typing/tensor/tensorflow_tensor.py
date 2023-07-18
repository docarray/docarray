from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar, Union, cast

import numpy as np

from docarray.base_doc.base_node import BaseNode
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import (
    import_library,
    is_jax_available,
    is_torch_available,
)

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.computation.tensorflow_backend import TensorFlowCompBackend
    from docarray.proto import NdArrayProto
else:
    tf = import_library('tensorflow', raise_error=True)

torch_available = is_torch_available()
if torch_available:
    import torch

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

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
    TensorFlowTensor class with a `.tensor` attribute of type `tf.Tensor`,
    intended for use in a Document.

    This enables (de)serialization from/to protobuf and json, data validation,
    and coercion from compatible types like numpy.ndarray.

    This type can also be used in a parametrized way, specifying the shape of the
    tensor.

    In comparison to [`TorchTensor`][docarray.typing.TorchTensor] and
    [`NdArray`][docarray.typing.tensor.ndarray.NdArray],
    [`TensorFlowTensor`][docarray.typing.tensor.tensorflow_tensor.TensorFlowTensor]
    is not a subclass of `tf.Tensor` (or `torch.Tensor`, `np.ndarray` respectively).
    Instead, the `tf.Tensor` is stored in
    [`TensorFlowTensor.tensor`][docarray.typing.tensor.tensorflow_tensor.TensorFlowTensor].
    Therefore, to do operations on the actual tensor data you have to always access the
    [`TensorFlowTensor.tensor`][docarray.typing.tensor.tensorflow_tensor.TensorFlowTensor]
    attribute.

    ---

    ```python
    import tensorflow as tf
    from docarray.typing import TensorFlowTensor


    t = TensorFlowTensor(tensor=tf.zeros((224, 224)))

    # tensorflow functions
    broadcasted = tf.broadcast_to(t.tensor, (3, 224, 224))
    broadcasted = tf.broadcast_to(t.unwrap(), (3, 224, 224))

    # this will fail:
    # broadcasted = tf.broadcast_to(t, (3, 224, 224))

    # tensorflow.Tensor methods:
    arr = t.tensor.numpy()
    arr = t.unwrap().numpy()

    # this will fail:
    # arr = t.numpy()
    ```

    ---

    The [`TensorFlowBackend`] however, operates on our
    [`TensorFlowTensor`][docarray.typing.TensorFlowTensor] instances.
    Here, you do not have to access the `.tensor` attribute,
    but can instead just hand over your
    [`TensorFlowTensor`][docarray.typing.TensorFlowTensor] instance.

    ---

    ```python
    import tensorflow as tf
    from docarray.typing import TensorFlowTensor


    zeros = TensorFlowTensor(tensor=tf.zeros((3, 224, 224)))

    comp_be = zeros.get_comp_backend()
    reshaped = comp_be.reshape(zeros, (224, 224, 3))
    assert comp_be.shape(reshaped) == (224, 224, 3)
    ```

    ---

    You can use [`TensorFlowTensor`][docarray.typing.TensorFlowTensor] in a Document as follows:

    ---

    ```python
    from docarray import BaseDoc
    from docarray.typing import TensorFlowTensor
    import tensorflow as tf


    class MyDoc(BaseDoc):
        tensor: TensorFlowTensor
        image_tensor: TensorFlowTensor[3, 224, 224]
        square_crop: TensorFlowTensor[3, 'x', 'x']
        random_image: TensorFlowTensor[
            3, ...
        ]  # first dimension is fixed, can have arbitrary shape


    # create a document with tensors
    doc = MyDoc(
        tensor=tf.zeros((128,)),
        image_tensor=tf.zeros((3, 224, 224)),
        square_crop=tf.zeros((3, 64, 64)),
        random_image=tf.zeros((3, 128, 256)),
    )

    # automatic shape conversion
    doc = MyDoc(
        tensor=tf.zeros((128,)),
        image_tensor=tf.zeros((224, 224, 3)),  # will reshape to (3, 224, 224)
        square_crop=tf.zeros((3, 128, 128)),
        random_image=tf.zeros((3, 64, 128)),
    )

    # !! The following will raise an error due to shape mismatch !!
    from pydantic import ValidationError

    try:
        doc = MyDoc(
            tensor=tf.zeros((128,)),
            image_tensor=tf.zeros((224, 224)),  # this will fail validation
            square_crop=tf.zeros((3, 128, 64)),  # this will also fail validation
            random_image=tf.zeros(4, 64, 128),  # this will also fail validation
        )
    except ValidationError as e:
        pass
    ```

    ---
    """

    __parametrized_meta__ = metaTensorFlow

    def __init__(self, tensor: tf.Tensor):
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, item):
        from docarray.computation.tensorflow_backend import TensorFlowCompBackend

        tensor = self.unwrap()
        if tensor is not None:
            tensor = tensor[item]
        return TensorFlowCompBackend._cast_output(t=tensor)

    def __setitem__(self, index, value):
        """Set a slice of this tensor's `tf.Tensor`"""
        t = self.unwrap()
        value = tf.cast(value, dtype=t.dtype)
        var = tf.Variable(t)
        var[index].assign(value)
        self.tensor = tf.constant(var)

    def __iter__(self):
        """Iterate over the elements of this tensor's `tf.Tensor`."""
        for i in range(len(self)):
            yield self[i]

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
        elif isinstance(value, np.ndarray):
            return cls._docarray_from_ndarray(value)
        elif isinstance(value, AbstractTensor):
            return cls._docarray_from_ndarray(value._docarray_to_ndarray())
        elif torch_available and isinstance(value, torch.Tensor):
            return cls._docarray_from_native(value.detach().cpu().numpy())
        elif jax_available and isinstance(value, jnp.ndarray):
            return cls._docarray_from_native(value.__array__())
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
        Create a `TensorFlowTensor` from a `tf.Tensor` or `TensorFlowTensor`
        instance.

        :param value: instance of `tf.Tensor` or `TensorFlowTensor`
        :return: a `TensorFlowTensor`
        """
        if isinstance(value, TensorFlowTensor):
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
                value.__class__ = cls.__unparametrizedcls__  # type: ignore
            else:
                value.__class__ = cls
            return cast(T, value)
        else:
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
                cls_param_ = cls.__unparametrizedcls__
                cls_param = cast(Type[T], cls_param_)
            else:
                cls_param = cls

            return cls_param(tensor=value)

    @staticmethod
    def get_comp_backend() -> 'TensorFlowCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.tensorflow_backend import TensorFlowCompBackend

        return TensorFlowCompBackend()

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert `TensorFlowTensor` into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.unwrap().numpy()

    def to_protobuf(self) -> 'NdArrayProto':
        """
        Transform self into an NdArrayProto protobuf message.
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        value_np = self.tensor.numpy()
        nd_proto.dense.buffer = value_np.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(value_np.shape))
        nd_proto.dense.dtype = value_np.dtype.str

        return nd_proto

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        Read ndarray from a proto msg.
        :param pb_msg:
        :return: a `TensorFlowTensor`
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls.from_ndarray(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.from_ndarray(np.zeros(source.shape))
        else:
            raise ValueError(
                f'Proto message {pb_msg} cannot be cast to a TensorFlowTensor.'
            )

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `TensorFlowTensor` from a numpy array.

        :param value: the numpy array
        :return: a `TensorFlowTensor`
        """
        return cls._docarray_from_native(tf.convert_to_tensor(value))

    def unwrap(self) -> tf.Tensor:
        """
        Return the original `tf.Tensor` without any memory copy.

        The original view rest intact and is still a Document `TensorFlowTensor`
        but the return object is a pure `tf.Tensor` but both object share
        the same memory layout.

        ---

        ```python
        from docarray.typing import TensorFlowTensor
        import tensorflow as tf

        t1 = TensorFlowTensor.validate(tf.zeros((3, 224, 224)), None, None)
        # here t1 is a docarray TensorFlowTensor
        t2 = t1.unwrap()
        # here t2 is a pure tf.Tensor but t1 is still a Docarray TensorFlowTensor
        ```

        ---
        :return: a `tf.Tensor`
        """
        return self.tensor

    def __len__(self) -> int:
        return len(self.tensor)

    @classmethod
    def _docarray_from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `tensor from a numpy array
        PS: this function is different from `from_ndarray` because it is private under the docarray namesapce.
        This allows us to avoid breaking change if one day we introduce a Tensor backend with a `from_ndarray` method.
        """
        return cls.from_ndarray(value)

    def _docarray_to_ndarray(self) -> np.ndarray:
        """cast itself to a numpy array"""
        return self.tensor.numpy()

    @property
    def shape(self):
        return tf.shape(self.tensor)
