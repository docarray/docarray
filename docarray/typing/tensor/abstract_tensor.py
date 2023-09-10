import abc
import warnings
from abc import ABC
from functools import reduce
from operator import mul
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray.base_doc.io.json import orjson_dumps
from docarray.computation import AbstractComputationalBackend
from docarray.typing.abstract_type import AbstractType
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic_core import CoreSchema, core_schema

if TYPE_CHECKING:
    from docarray.proto import NdArrayProto, NodeProto

T = TypeVar('T', bound='AbstractTensor')
TTensor = TypeVar('TTensor')
ShapeT = TypeVar('ShapeT')


# displaying tensors that are too large causes problems in the browser
DISPLAY_TENSOR_OPENAPI_MAX_ITEMS = 256


class _ParametrizedMeta(type):
    """
    This metaclass ensures that instance, subclass and equality checks on parametrized Tensors
    are handled as expected:

    assert safe_issubclass(TorchTensor[128], TorchTensor[128])
    t = parse_obj_as(TorchTensor[128], torch.zeros(128))
    assert isinstance(t, TorchTensor[128])
    TorchTensor[128] == TorchTensor[128]
    hash(TorchTensor[128]) == hash(TorchTensor[128])
    etc.

    This special handling is needed because every call to `AbstractTensor.__getitem__`
    creates a new class on the fly.
    We want technically distinct but identical classes to be considered equal.
    """

    def _equals_special_case(cls, other):
        is_type = isinstance(other, type)
        is_tensor = is_type and AbstractTensor in other.__mro__
        same_parents = is_tensor and cls.__mro__[1:] == other.__mro__[1:]

        subclass_target_shape = getattr(other, '__docarray_target_shape__', False)
        self_target_shape = getattr(cls, '__docarray_target_shape__', False)
        same_shape = (
            same_parents
            and subclass_target_shape
            and self_target_shape
            and subclass_target_shape == self_target_shape
        )

        return same_shape

    def __subclasscheck__(cls, subclass):
        if cls._equals_special_case(subclass):
            return True
        return super().__subclasscheck__(subclass)

    def __instancecheck__(cls, instance):
        is_tensor = isinstance(instance, AbstractTensor)
        if is_tensor:  # custom handling
            _cls = cast(Type[AbstractTensor], cls)
            if (
                _cls.__unparametrizedcls__
            ):  # This is not None if the tensor is parametrized
                if (
                    instance.get_comp_backend().shape(instance)
                    != _cls.__docarray_target_shape__
                ):
                    return False
                return any(
                    safe_issubclass(candidate, _cls.__unparametrizedcls__)
                    for candidate in type(instance).__mro__
                )
            return any(
                safe_issubclass(candidate, cls) for candidate in type(instance).__mro__
            )
        return super().__instancecheck__(instance)

    def __eq__(cls, other):
        if cls._equals_special_case(other):
            return True
        return NotImplemented

    def __hash__(cls):
        try:
            cls_ = cast(AbstractTensor, cls)
            return hash((cls_.__docarray_target_shape__, cls_.__unparametrizedcls__))
        except AttributeError:
            raise NotImplementedError(
                '`hash()` is not implemented for this class. The `_ParametrizedMeta` '
                'metaclass should only be used for `AbstractTensor` subclasses. '
                'Otherwise, you have to implement `__hash__` for your class yourself.'
            )


class AbstractTensor(Generic[TTensor, T], AbstractType, ABC, Sized):
    __parametrized_meta__: type = _ParametrizedMeta
    __unparametrizedcls__: Optional[Type['AbstractTensor']] = None
    __docarray_target_shape__: Optional[Tuple[int, ...]] = None
    _proto_type_name: str

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert itself into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(ndarray=nd_proto, type=self._proto_type_name)

    @classmethod
    def __docarray_validate_shape__(cls, t: T, shape: Tuple[Union[int, str], ...]) -> T:
        """Every tensor has to implement this method in order to
        enable syntax of the form AnyTensor[shape].
        It is called when a tensor is assigned to a field of this type.
        i.e. when a tensor is passed to a Document field of type AnyTensor[shape].

        The intended behaviour is as follows:

        - If the shape of `t` is equal to `shape`, return `t`.
        - If the shape of `t` is not equal to `shape`,
            but can be reshaped to `shape`, return `t` reshaped to `shape`.
        - If the shape of `t` is not equal to `shape`
            and cannot be reshaped to `shape`, raise a ValueError.

        :param t: The tensor to validate.
        :param shape: The shape to validate against.
        :return: The validated tensor.
        """
        comp_be = t.get_comp_backend()
        tshape = comp_be.shape(t)
        if tshape == shape:
            return t
        elif any(isinstance(dim, str) or dim == Ellipsis for dim in shape):
            ellipsis_occurrences = [
                pos for pos, dim in enumerate(shape) if dim == Ellipsis
            ]
            if ellipsis_occurrences:
                if len(ellipsis_occurrences) > 1:
                    raise ValueError(
                        f'Cannot use Ellipsis (...) more than once for the shape {shape}'
                    )
                ellipsis_pos = ellipsis_occurrences[0]
                # Calculate how many dimensions to add. Should be at least 1.
                dimensions_needed = max(len(tshape) - len(shape) + 1, 1)
                shape = (
                    shape[:ellipsis_pos]
                    + tuple(
                        f'__dim_var_{index}__' for index in range(dimensions_needed)
                    )
                    + shape[ellipsis_pos + 1 :]
                )

            if len(tshape) != len(shape):
                raise ValueError(
                    f'Tensor shape mismatch. Expected {shape}, got {tshape}'
                )
            known_dims: Dict[str, int] = {}
            for tdim, dim in zip(tshape, shape):
                if isinstance(dim, int) and tdim != dim:
                    raise ValueError(
                        f'Tensor shape mismatch. Expected {shape}, got {tshape}'
                    )
                elif isinstance(dim, str):
                    if dim in known_dims and known_dims[dim] != tdim:
                        raise ValueError(
                            f'Tensor shape mismatch. Expected {shape}, got {tshape}'
                        )
                    else:
                        known_dims[dim] = tdim
            else:
                return t
        else:
            shape = cast(Tuple[int], shape)
            warnings.warn(
                f'Tensor shape mismatch. Reshaping tensor '
                f'of shape {tshape} to shape {shape}'
            )
            try:
                value = cls._docarray_from_native(comp_be.reshape(t, shape))
                return cast(T, value)
            except RuntimeError:
                raise ValueError(
                    f'Cannot reshape tensor of shape {tshape} to shape {shape}'
                )

    @classmethod
    def __docarray_validate_getitem__(cls, item: Any) -> Tuple[int]:
        """This method validates the input to `AbstractTensor.__class_getitem__`.

        It is called at "class creation time",
        i.e. when a class is created with syntax of the form AnyTensor[shape].

        The default implementation tries to cast any `item` to a tuple of ints.
        A subclass can override this method to implement custom validation logic.

        The output of this is eventually passed to
        [`AbstractTensor.__docarray_validate_shape__`]
        [docarray.typing.tensor.abstract_tensor.AbstractTensor.__docarray_validate_shape__]
        as its `shape` argument.

        Raises `ValueError` if the input `item` does not pass validation.

        :param item: The item to validate, passed to `__class_getitem__` (`Tensor[item]`).
        :return: The validated item == the target shape of this tensor.
        """
        if isinstance(item, int):
            item = (item,)
        try:
            item = tuple(item)
        except TypeError:
            raise TypeError(f'{item} is not a valid tensor shape.')
        return item

    if is_pydantic_v2:

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
        ) -> Dict[str, Any]:
            json_schema = {}
            json_schema.update(type='array', items={'type': 'number'})
            if cls.__docarray_target_shape__ is not None:
                shape_info = (
                    '['
                    + ', '.join([str(s) for s in cls.__docarray_target_shape__])
                    + ']'
                )
                if (
                    reduce(mul, cls.__docarray_target_shape__, 1)
                    <= DISPLAY_TENSOR_OPENAPI_MAX_ITEMS
                ):
                    # custom example only for 'small' shapes, otherwise it is too big to display
                    example_payload = orjson_dumps(
                        np.zeros(cls.__docarray_target_shape__)
                    ).decode()
                    json_schema.update(example=example_payload)
            else:
                shape_info = 'not specified'
            json_schema['tensor/array shape'] = shape_info
            return json_schema

    else:

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='array', items={'type': 'number'})
            if cls.__docarray_target_shape__ is not None:
                shape_info = (
                    '['
                    + ', '.join([str(s) for s in cls.__docarray_target_shape__])
                    + ']'
                )
                if (
                    reduce(mul, cls.__docarray_target_shape__, 1)
                    <= DISPLAY_TENSOR_OPENAPI_MAX_ITEMS
                ):
                    # custom example only for 'small' shapes, otherwise it is too big to display
                    example_payload = orjson_dumps(
                        np.zeros(cls.__docarray_target_shape__)
                    ).decode()
                    field_schema.update(example=example_payload)
            else:
                shape_info = 'not specified'
            field_schema['tensor/array shape'] = shape_info

    @classmethod
    def _docarray_create_parametrized_type(cls: Type[T], shape: Tuple[int]):
        shape_str = ', '.join([str(s) for s in shape])

        class _ParametrizedTensor(
            cls,  # type: ignore
            metaclass=cls.__parametrized_meta__,  # type: ignore
        ):
            __unparametrizedcls__ = cls
            __docarray_target_shape__ = shape

            @classmethod
            def _docarray_validate(
                _cls,
                value: Any,
            ):
                t = super()._docarray_validate(value)
                return _cls.__docarray_validate_shape__(
                    t, _cls.__docarray_target_shape__
                )

        _ParametrizedTensor.__name__ = f'{cls.__name__}[{shape_str}]'
        _ParametrizedTensor.__qualname__ = f'{cls.__qualname__}[{shape_str}]'

        return _ParametrizedTensor

    def __class_getitem__(cls, item: Any):
        target_shape = cls.__docarray_validate_getitem__(item)
        return cls._docarray_create_parametrized_type(target_shape)

    @classmethod
    def _docarray_stack(cls: Type[T], seq: Union[List[T], Tuple[T]]) -> T:
        """Stack a sequence of tensors into a single tensor."""
        comp_backend = cls.get_comp_backend()
        # at runtime, 'T' is always the correct input type for .stack()
        # but mypy doesn't know that, so we ignore it here
        return cls._docarray_from_native(comp_backend.stack(seq))  # type: ignore

    @classmethod
    @abc.abstractmethod
    def _docarray_from_native(cls: Type[T], value: Any) -> T:
        """
        Create a DocList tensor from a tensor that is native to the given framework,
        e.g. from numpy.ndarray or torch.Tensor.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def get_comp_backend() -> AbstractComputationalBackend:
        """The computational backend compatible with this tensor type."""
        ...

    @abc.abstractmethod
    def __getitem__(self: T, item) -> T:
        """Get a slice of this tensor."""
        ...

    @abc.abstractmethod
    def __setitem__(self, index, value):
        """Set a slice of this tensor."""
        ...

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over the elements of this tensor."""
        ...

    @abc.abstractmethod
    def to_protobuf(self) -> 'NdArrayProto':
        """Convert DocList into a Protobuf message"""
        ...

    def unwrap(self):
        """Return the native tensor object that this DocList tensor wraps."""

    @abc.abstractmethod
    def _docarray_to_json_compatible(self):
        """
        Convert tensor into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self

    @classmethod
    @abc.abstractmethod
    def _docarray_from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a `tensor from a numpy array
        PS: this function is different from `from_ndarray` because it is private under the docarray namesapce.
        This allows us to avoid breaking change if one day we introduce a Tensor backend with a `from_ndarray` method.
        """
        ...

    @abc.abstractmethod
    def _docarray_to_ndarray(self) -> np.ndarray:
        """cast itself to a numpy array"""
        ...

    if is_pydantic_v2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.general_plain_validator_function(
                cls.validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    function=orjson_dumps,
                    return_schema=handler.generate_schema(bytes),
                    when_used="json-unless-none",
                ),
            )
