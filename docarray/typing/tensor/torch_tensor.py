from copy import copy
from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar, Union, cast

import numpy as np
import orjson

from docarray.base_doc.base_node import BaseNode
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import (
    import_library,
    is_jax_available,
    is_tf_available,
)

if TYPE_CHECKING:
    import torch

    from docarray.computation.torch_backend import TorchCompBackend
    from docarray.proto import NdArrayProto
else:
    torch = import_library('torch', raise_error=True)

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

T = TypeVar('T', bound='TorchTensor')
ShapeT = TypeVar('ShapeT')

torch_base: type = type(torch.Tensor)
node_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaTorchAndNode(
    AbstractTensor.__parametrized_meta__,  # type: ignore
    torch_base,  # type: ignore
    node_base,  # type: ignore
):  # type: ignore
    pass


@_register_proto(proto_type_name='torch_tensor')
class TorchTensor(
    torch.Tensor,
    AbstractTensor,
    Generic[ShapeT],
    metaclass=metaTorchAndNode,
):
    # Subclassing torch.Tensor following the advice from here:
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    """
    Subclass of `torch.Tensor`, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coercion from compatible types like numpy.ndarray.

    This type can also be used in a parametrized way,
    specifying the shape of the tensor.

    ---

    ```python
    from docarray import BaseDoc
    from docarray.typing import TorchTensor
    import torch


    class MyDoc(BaseDoc):
        tensor: TorchTensor
        image_tensor: TorchTensor[3, 224, 224]
        square_crop: TorchTensor[3, 'x', 'x']
        random_image: TorchTensor[
            3, ...
        ]  # first dimension is fixed, can have arbitrary shape


    # create a document with tensors
    doc = MyDoc(
        tensor=torch.zeros(128),
        image_tensor=torch.zeros(3, 224, 224),
        square_crop=torch.zeros(3, 64, 64),
        random_image=torch.zeros(3, 128, 256),
    )

    # automatic shape conversion
    doc = MyDoc(
        tensor=torch.zeros(128),
        image_tensor=torch.zeros(224, 224, 3),  # will reshape to (3, 224, 224)
        square_crop=torch.zeros(3, 128, 128),
        random_image=torch.zeros(3, 64, 128),
    )

    # !! The following will raise an error due to shape mismatch !!
    from pydantic import ValidationError

    try:
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(224, 224),  # this will fail validation
            square_crop=torch.zeros(3, 128, 64),  # this will also fail validation
            random_image=torch.zeros(4, 64, 128),  # this will also fail validation
        )
    except ValidationError as e:
        pass
    ```

    ---


    ## Compatibility with `torch.compile()`


    PyTorch 2 [introduced compilation support](https://pytorch.org/blog/pytorch-2.0-release/) in the form of `torch.compile()`.

    Currently, **`torch.compile()` does not properly support subclasses of `torch.Tensor` such as `TorchTensor`**.
    The PyTorch team is currently working on a [fix for this issue](https://github.com/pytorch/pytorch/pull/105167#issuecomment-1678050808).

    In the meantime, you can use the following workaround:

    ### Workaround: Convert `TorchTensor` to `torch.Tensor` before calling `torch.compile()`

    Converting any `TorchTensor`s tor `torch.Tensor` before calling `torch.compile()` side-steps the issue:

    ```python
    from docarray import BaseDoc
    from docarray.typing import TorchTensor
    import torch


    class MyDoc(BaseDoc):
        tensor: TorchTensor


    doc = MyDoc(tensor=torch.zeros(128))


    def foo(tensor: torch.Tensor):
        return tensor @ tensor.t()


    foo_compiled = torch.compile(foo)

    # unwrap the tensor before passing it to torch.compile()
    foo_compiled(doc.tensor.unwrap())
    ```

    """

    __parametrized_meta__ = metaTorchAndNode

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Union[T, np.ndarray, str, Any],
    ) -> T:
        if isinstance(value, TorchTensor):
            return cast(T, value)
        elif isinstance(value, torch.Tensor):
            return cls._docarray_from_native(value)
        elif isinstance(value, AbstractTensor):
            return cls._docarray_from_ndarray(value._docarray_to_ndarray())
        elif tf_available and isinstance(value, tf.Tensor):
            return cls._docarray_from_ndarray(value.numpy())
        elif isinstance(value, np.ndarray):
            return cls._docarray_from_ndarray(value)
        elif jax_available and isinstance(value, jnp.ndarray):
            return cls._docarray_from_ndarray(value.__array__())
        elif isinstance(value, str):
            value = orjson.loads(value)
        try:
            arr: torch.Tensor = torch.tensor(value)
            return cls._docarray_from_native(arr)
        except Exception:
            pass  # handled below

        raise ValueError(f'Expected a torch.Tensor compatible type, got {type(value)}')

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert `TorchTensor` into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.detach().numpy()  # might need to check device later

    def unwrap(self) -> torch.Tensor:
        """
        Return the original `torch.Tensor` without any memory copy.

        The original view rest intact and is still a Document `TorchTensor`
        but the return object is a pure `torch.Tensor` but both object share
        the same memory layout.

        ---

        ```python
        from docarray.typing import TorchTensor
        import torch
        from pydantic import parse_obj_as


        t = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))
        # here t is a docarray TorchTensor
        t2 = t.unwrap()
        # here t2 is a pure torch.Tensor but t1 is still a Docarray TorchTensor
        # But both share the same underlying memory
        ```

        ---

        :return: a `torch.Tensor`
        """
        value = copy(self)  # as unintuitive as it sounds, this
        # does not do any relevant memory copying, just shallow
        # reference to the torch data
        value.__class__ = torch.Tensor  # type: ignore
        return value

    @classmethod
    def _docarray_from_native(cls: Type[T], value: torch.Tensor) -> T:
        """Create a `TorchTensor` from a native `torch.Tensor`

        :param value: the native `torch.Tensor`
        :return: a `TorchTensor`
        """
        if cls.__unparametrizedcls__:  # This is not None if the tensor is parametrized
            value.__class__ = cls.__unparametrizedcls__  # type: ignore
        else:
            value.__class__ = cls
        return cast(T, value)

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `TorchTensor` from a numpy array

        :param value: the numpy array
        :return: a `TorchTensor`
        """
        return cls._docarray_from_native(torch.from_numpy(value))

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        Read ndarray from a proto msg
        :param pb_msg:
        :return: a `TorchTensor`
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
        Transform self into a `NdArrayProto` protobuf message
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
    def get_comp_backend() -> 'TorchCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.torch_backend import TorchCompBackend

        return TorchCompBackend()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # this tells torch to treat all of our custom tensors just like
        # torch.Tensor's. Otherwise, torch will complain that it doesn't
        # know how to handle our custom tensor type.
        docarray_torch_tensors = TorchTensor.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in docarray_torch_tensors else t for t in types
        )
        return super().__torch_function__(func, types_, args, kwargs)

    def __deepcopy__(self, memo):
        """
        Custom implementation of deepcopy for TorchTensor to avoid storage sharing issues.
        """
        # Create a new tensor with the same data and properties
        new_tensor = self.clone()
        # Set the class to the custom TorchTensor class
        new_tensor.__class__ = self.__class__
        return new_tensor

    @classmethod
    def _docarray_from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `tensor from a numpy array
        PS: this function is different from `from_ndarray` because it is private under the docarray namesapce.
        This allows us to avoid breaking change if one day we introduce a Tensor backend with a `from_ndarray` method.
        """
        return cls.from_ndarray(value)

    def _docarray_to_ndarray(self) -> np.ndarray:
        """cast itself to a numpy array"""
        return self.detach().cpu().numpy()

    def new_empty(self, *args, **kwargs):
        """
        This method enables the deepcopy of `TorchTensor` by returning another instance of this subclass.
        If this function is not implemented, the deepcopy will throw an RuntimeError from Torch.
        """
        return self.__class__(*args, **kwargs)
