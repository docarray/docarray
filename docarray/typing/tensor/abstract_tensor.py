import abc
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, Tuple, Type, TypeVar

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
        enbale syntax of the form Tensor[shape].
        The intended behavoiour is as follows:
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

    def __class_getitem__(cls, item):
        if isinstance(item, int):
            item = (item,)
        try:
            item = tuple(item)
        except TypeError:
            raise TypeError(f'{item} is not a valid tensor shape.')

        return cls._create_parametrized_type(item)
