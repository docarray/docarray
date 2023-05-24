from typing import TYPE_CHECKING, Any, Generic, List, Tuple, Type, TypeVar, Union, cast

import jax.numpy as jnp

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.computation.jax_backend import JaxCompBackend
    from docarray.proto import NdArrayProto

from docarray.base_doc.base_node import BaseNode

T = TypeVar('T', bound='JaxArray')
ShapeT = TypeVar('ShapeT')

tensor_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaNumpy(AbstractTensor.__parametrized_meta__, tensor_base):  # type: ignore
    pass


@_register_proto(proto_type_name='jaxarray')
class JaxArray(jnp.ndarray, AbstractTensor, Generic[ShapeT]):
    """ """

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
        value: Union[T, jnp.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, jnp.ndarray):
            return cls._docarray_from_native(value)
        elif isinstance(value, JaxArray):
            return cast(T, value)
        elif isinstance(value, list) or isinstance(value, tuple):
            try:
                arr_from_list: jnp.ndarray = jnp.asarray(value)
                return cls._docarray_from_native(arr_from_list)
            except Exception:
                pass  # handled below
        else:
            try:
                arr: jnp.ndarray = jnp.ndarray(value)
                return cls._docarray_from_native(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a numpy.ndarray compatible type, got {type(value)}')

    @classmethod
    def _docarray_from_native(cls: Type[T], value: jnp.ndarray) -> T:
        if cls.__unparametrizedcls__:  # This is not None if the tensor is parametrized
            return cast(T, value.view(cls.__unparametrizedcls__))
        return value.view(cls)

    def _docarray_to_json_compatible(self) -> jnp.ndarray:
        """
        Convert `JaxArray` into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.unwrap()

    def unwrap(self) -> jnp.ndarray:
        """
        Return the original ndarray without any memory copy.

        The original view rest intact and is still a Document `JaxArray`
        but the return object is a pure `np.ndarray` but both object share
        the same memory layout.

        ---

        ```python
        from docarray.typing import JaxArray
        import numpy as np

        t1 = JaxArray.validate(np.zeros((3, 224, 224)), None, None)
        # here t1 is a docarray NdArray
        t2 = t1.unwrap()
        # here t2 is a pure np.ndarray but t1 is still a Docarray JaxArray
        # But both share the same underlying memory
        ```

        ---

        :return: a `jnp.ndarray`
        """
        return self.view(jnp.ndarray)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        Read ndarray from a proto msg
        :param pb_msg:
        :return: a numpy array
        """
        source = pb_msg.dense
        if source.buffer:
            x = jnp.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls._docarray_from_native(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls._docarray_from_native(jnp.zeros(source.shape))
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
    def get_comp_backend() -> 'JaxCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.jax_backend import JaxCompBackend

        return JaxCompBackend()

    def __class_getitem__(cls, item: Any, *args, **kwargs):
        # see here for mypy bug: https://github.com/python/mypy/issues/14123
        return AbstractTensor.__class_getitem__.__func__(cls, item)  # type: ignore
