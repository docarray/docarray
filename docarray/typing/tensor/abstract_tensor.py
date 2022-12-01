import abc
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, Sequence, Tuple, Type, TypeVar

from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='AbstractTensor')
ShapeT = TypeVar('ShapeT')


class AbstractTensor(AbstractType, Generic[ShapeT], ABC):

    __parametrized_meta__ = type

    @classmethod
    @abc.abstractmethod
    def __validate_shape__(cls, t: T, shape: Tuple[int]) -> T:
        """Every tensor has to implement this method in order to
        enable syntax of the form Tensor[shape].

        It is called when a tensor is assigned to a field of this type.
        i.e. when a tensor is passed to a Document field of type Tensor[shape].

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
        ...

    @classmethod
    def __validate_getitem__(cls, item: Any) -> Tuple[int]:
        """This method validates the input to __class_getitem__.

        It is called at "class creation time",
        i.e. when a class is created with syntax of the form Tensor[shape].

        The default implementation tries to cast any `item` to a tuple of ints.
        A subclass can override this method to implement custom validation logic.

        The output of this is eventually passed to
        {ref}`AbstractTensor.__validate_shape__` as its `shape` argument.

        Raises `ValueError` if the input `item` does not pass validation.

        :param item: The item to validate, passed to __class_getitem__ (`Tensor[item]`).
        :return: The validated item == the target shape of this tensor.
        """
        if isinstance(item, int):
            item = (item,)
        try:
            item = tuple(item)
        except TypeError:
            raise TypeError(f'{item} is not a valid tensor shape.')
        return item

    @classmethod
    def _create_parametrized_type(cls: Type[T], shape: Tuple[int]):
        shape_str = ', '.join([str(s) for s in shape])

        class _ParametrizedTensor(
            cls,  # type: ignore
            metaclass=cls.__parametrized_meta__,  # type: ignore
        ):
            _docarray_target_shape = shape
            __name__ = f'{cls.__name__}[{shape_str}]'
            __qualname__ = f'{cls.__qualname__}[{shape_str}]'

            @classmethod
            def validate(
                _cls,
                value: Any,
                field: 'ModelField',
                config: 'BaseConfig',
            ):
                t = super().validate(value, field, config)
                return _cls.__validate_shape__(t, _cls._docarray_target_shape)

        return _ParametrizedTensor

<<<<<<< HEAD
    def __class_getitem__(cls, item: Any):
        target_shape = cls.__validate_getitem__(item)
        return cls._create_parametrized_type(target_shape)

=======
>>>>>>> feat: embedding type (#877)
    @classmethod
    def __docarray_stack__(cls, seq: Sequence[T]) -> T:
        """Stack a sequence of tensors into a single tensor."""
        ...
    def __class_getitem__(cls, item: Any):
        target_shape = cls.__validate_getitem__(item)
        return cls._create_parametrized_type(target_shape)
