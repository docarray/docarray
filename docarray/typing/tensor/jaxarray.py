from typing import TYPE_CHECKING, Any, Generic, List, Tuple, Type, TypeVar, Union, cast

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.computation.jax_backend import JaxCompBackend
    from docarray.proto import NdArrayProto
else:
    jax = import_library('jax', raise_error=True)
    jnp = jax.numpy
from docarray.base_doc.base_node import BaseNode

T = TypeVar('T', bound='JaxArray')
ShapeT = TypeVar('ShapeT')

node_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaJax(
    AbstractTensor.__parametrized_meta__,  # type: ignore
    node_base,  # type: ignore
):  # type: ignore
    pass


@_register_proto(proto_type_name='jaxarray')
class JaxArray(AbstractTensor, Generic[ShapeT], metaclass=metaJax):
    """
    Subclass of `jnp.ndarray`, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coercion from compatible types like `torch.Tensor`.

    This type can also be used in a parametrized way, specifying the shape of the array.

    ---

    ```python
    from docarray import BaseDoc
    from docarray.typing import JaxArray
    import jax.numpy as jnp


    class MyDoc(BaseDoc):
        arr: JaxArray
        image_arr: JaxArray[3, 224, 224]
        square_crop: JaxArray[3, 'x', 'x']
        random_image: JaxArray[3, ...]  # first dimension is fixed, can have arbitrary shape


    # create a document with tensors
    doc = MyDoc(
        arr=jnp.zeros((128,)),
        image_arr=jnp.zeros((3, 224, 224)),
        square_crop=jnp.zeros((3, 64, 64)),
        random_image=jnp.zeros((3, 128, 256)),
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

    __parametrized_meta__ = metaJax

    def __init__(self, tensor: jnp.ndarray):
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, item):
        from docarray.computation.jax_backend import JaxCompBackend

        tensor = self.unwrap()
        if tensor is not None:
            tensor = tensor[item]
        return JaxCompBackend._cast_output(t=tensor)

    def __setitem__(self, index, value):
        """"""
        # print(index, value)
        self.tensor = self.tensor.at[index : index + 1].set(value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.tensor)

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
        if isinstance(value, jax.Array):
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
        if isinstance(value, JaxArray):
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
                value.__class__ = cls.__unparametrizedcls__  # type: ignore
            else:
                value.__class__ = cls  # type: ignore
            return cast(T, value)
        else:
            if cls.__unparametrizedcls__:  # None if the tensor is parametrized
                cls_param_ = cls.__unparametrizedcls__
                cls_param = cast(Type[T], cls_param_)
            else:
                cls_param = cls

            return cls_param(tensor=value)

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `TensorFlowTensor` from a numpy array.

        :param value: the numpy array
        :return: a `TensorFlowTensor`
        """
        return cls._docarray_from_native(jnp.array(value))

    def _docarray_to_json_compatible(self) -> jnp.ndarray:
        """
        Convert `JaxArray` into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.unwrap()

    def unwrap(self) -> jnp.ndarray:
        """
        Return the original ndarray without making a copy in memory.

        The original view remains intact and is still a Document `JaxArray`
        but the return object is a pure `np.ndarray` and both objects share
        the same underlying memory.

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
        return self.tensor

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
            return cls.from_ndarray(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.from_ndarray(np.zeros(source.shape))
        else:
            raise ValueError(
                f'Proto message {pb_msg} cannot be cast to a TensorFlowTensor.'
            )

    def to_protobuf(self) -> 'NdArrayProto':
        """
        Transform self into a NdArrayProto protobuf message
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        value_np = self.tensor
        nd_proto.dense.buffer = value_np.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(value_np.shape))
        nd_proto.dense.dtype = value_np.dtype.str

        return nd_proto

    @staticmethod
    def get_comp_backend() -> 'JaxCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.jax_backend import JaxCompBackend

        return JaxCompBackend()

    def __class_getitem__(cls, item: Any, *args, **kwargs):
        # see here for mypy bug: https://github.com/python/mypy/issues/14123
        return AbstractTensor.__class_getitem__.__func__(cls, item)  # type: ignore

    @classmethod
    def _docarray_from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        return cls.from_ndarray(value)

    def _docarray_to_ndarray(self) -> np.ndarray:
        """cast itself to a numpy array"""
        return self.tensor.__array__()
