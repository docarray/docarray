from typing import TYPE_CHECKING, Any, Generic, List, Tuple, Type, TypeVar, Union, cast

import numpy as np

from docarray.base_doc.base_node import BaseNode
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import (  # noqa
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

    from docarray.typing.tensor.jaxarray import JaxArray  # noqa: F401

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.computation.numpy_backend import NumpyCompBackend
    from docarray.proto import NdArrayProto


T = TypeVar('T', bound='NdArray')
ShapeT = TypeVar('ShapeT')

tensor_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaNumpy(AbstractTensor.__parametrized_meta__, tensor_base):  # type: ignore
    pass


@_register_proto(proto_type_name='ndarray')
class NdArray(np.ndarray, AbstractTensor, Generic[ShapeT]):
    """
    Subclass of `np.ndarray`, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coercion from compatible types like `torch.Tensor`.

    This type can also be used in a parametrized way, specifying the shape of the array.

    ---

    ```python
    from docarray import BaseDoc
    from docarray.typing import NdArray
    import numpy as np


    class MyDoc(BaseDoc):
        arr: NdArray
        image_arr: NdArray[3, 224, 224]
        square_crop: NdArray[3, 'x', 'x']
        random_image: NdArray[3, ...]  # first dimension is fixed, can have arbitrary shape


    # create a document with tensors
    doc = MyDoc(
        arr=np.zeros((128,)),
        image_arr=np.zeros((3, 224, 224)),
        square_crop=np.zeros((3, 64, 64)),
        random_image=np.zeros((3, 128, 256)),
    )
    assert doc.image_arr.shape == (3, 224, 224)

    # automatic shape conversion
    doc = MyDoc(
        arr=np.zeros((128,)),
        image_arr=np.zeros((224, 224, 3)),  # will reshape to (3, 224, 224)
        square_crop=np.zeros((3, 128, 128)),
        random_image=np.zeros((3, 64, 128)),
    )
    assert doc.image_arr.shape == (3, 224, 224)

    # !! The following will raise an error due to shape mismatch !!
    from pydantic import ValidationError

    try:
        doc = MyDoc(
            arr=np.zeros((128,)),
            image_arr=np.zeros((224, 224)),  # this will fail validation
            square_crop=np.zeros((3, 128, 64)),  # this will also fail validation
            random_image=np.zeros((4, 64, 128)),  # this will also fail validation
        )
    except ValidationError as e:
        pass
    ```

    ---
    """

    __parametrized_meta__ = metaNumpy

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, np.ndarray):
            return cls._docarray_from_native(value)
        elif isinstance(value, NdArray):
            return cast(T, value)
        elif isinstance(value, AbstractTensor):
            return cls._docarray_from_native(value._docarray_to_ndarray())
        elif torch_available and isinstance(value, torch.Tensor):
            return cls._docarray_from_native(value.detach().cpu().numpy())
        elif tf_available and isinstance(value, tf.Tensor):
            return cls._docarray_from_native(value.numpy())
        elif jax_available and isinstance(value, jnp.ndarray):
            return cls._docarray_from_native(value.__array__())
        elif isinstance(value, list) or isinstance(value, tuple):
            try:
                arr_from_list: np.ndarray = np.asarray(value)
                return cls._docarray_from_native(arr_from_list)
            except Exception:
                pass  # handled below
        else:
            try:
                arr: np.ndarray = np.ndarray(value)
                return cls._docarray_from_native(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a numpy.ndarray compatible type, got {type(value)}')

    @classmethod
    def _docarray_from_native(cls: Type[T], value: np.ndarray) -> T:
        if cls.__unparametrizedcls__:  # This is not None if the tensor is parametrized
            return cast(T, value.view(cls.__unparametrizedcls__))
        return value.view(cls)

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert `NdArray` into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.unwrap()

    def unwrap(self) -> np.ndarray:
        """
        Return the original ndarray without any memory copy.

        The original view rest intact and is still a Document `NdArray`
        but the return object is a pure `np.ndarray` but both object share
        the same memory layout.

        ---

        ```python
        from docarray.typing import NdArray
        import numpy as np

        t1 = NdArray.validate(np.zeros((3, 224, 224)), None, None)
        # here t1 is a docarray NdArray
        t2 = t1.unwrap()
        # here t2 is a pure np.ndarray but t1 is still a Docarray NdArray
        # But both share the same underlying memory
        ```

        ---

        :return: a `numpy.ndarray`
        """
        return self.view(np.ndarray)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        Read ndarray from a proto msg
        :param pb_msg:
        :return: a numpy array
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls._docarray_from_native(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls._docarray_from_native(np.zeros(source.shape))
        else:
            raise ValueError(f'proto message {pb_msg} cannot be cast to a NdArray')

    def to_protobuf(self) -> 'NdArrayProto':
        """
        Transform self into a NdArrayProto protobuf message
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        nd_proto.dense.buffer = self.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(self.shape))
        nd_proto.dense.dtype = self.dtype.str

        return nd_proto

    @staticmethod
    def get_comp_backend() -> 'NumpyCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.numpy_backend import NumpyCompBackend

        return NumpyCompBackend()

    def __class_getitem__(cls, item: Any, *args, **kwargs):
        # see here for mypy bug: https://github.com/python/mypy/issues/14123
        return AbstractTensor.__class_getitem__.__func__(cls, item)  # type: ignore

    @classmethod
    def _docarray_from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `tensor from a numpy array
        PS: this function is different from `from_ndarray` because it is private under the docarray namesapce.
        This allows us to avoid breaking change if one day we introduce a Tensor backend with a `from_ndarray` method.
        """
        return cls._docarray_from_native(value)

    def _docarray_to_ndarray(self) -> np.ndarray:
        """Create a `tensor from a numpy array
        PS: this function is different from `from_ndarray` because it is private under the docarray namesapce.
        This allows us to avoid breaking change if one day we introduce a Tensor backend with a `from_ndarray` method.
        """
        return self.unwrap()
