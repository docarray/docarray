from abc import ABC
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar

from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='AbstractTensor')


def _get_attr_from_superclasses(supers, attr):
    for cls in supers:
        if hasattr(cls, attr):
            return getattr(cls, attr)
    raise AttributeError(f'Cannot find attribute {attr} in mro')


@classmethod
def __validate_parametrized__(
    cls: Type[T],
    value: Any,
    field: 'ModelField',
    config: 'BaseConfig',
) -> T:
    supers = cls.__mro__[1:]  # superclasses of cls
    _validate = _get_attr_from_superclasses(supers, 'validate')
    t = _validate(value, field, config)
    return cls.__validate_shape__(t, cls._docarray_target_shape)


class AbstractTensor(AbstractType, ABC):

    __parametrized_meta__ = type

    @classmethod
    def __validate_shape__(cls, t: T, shape: Tuple[int]) -> T:
        ...

    @classmethod
    def _create_parametrized_type(cls, shape: Tuple[int]):
        shape_str = ', '.join([str(s) for s in shape])
        return cls.__parametrized_meta__(
            f'{cls.__name__}[{shape_str}]',
            (cls,),
            {'_docarray_target_shape': shape, 'validate': __validate_parametrized__},
        )

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            if isinstance(item, int):
                item = (item,)
            else:
                raise TypeError(f'{item} is not a valid tensor shape.')

        return cls._create_parametrized_type(item)
