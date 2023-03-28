from typing import Any, Optional

from typing_inspect import get_args, is_optional_type, is_union_type

from docarray.typing.tensor.abstract_tensor import AbstractTensor


def is_type_tensor(type_: Any) -> bool:
    """Return True if type is a type Tensor or an Optional Tensor type."""
    return isinstance(type_, type) and issubclass(type_, AbstractTensor)


def is_tensor_union(type_: Any) -> bool:
    """Return True if type is a Union of type Tensors."""
    is_union = is_union_type(type_)
    if is_union is None:
        return False
    else:
        return is_union and all(
            (is_type_tensor(t) or issubclass(t, type(None))) for t in get_args(type_)
        )


def change_cls_name(cls: type, new_name: str, scope: Optional[dict] = None) -> None:
    """Change the name of a class.

    :param cls: the class to change the name of
    :param new_name: the new name
    :param scope: the scope in which the class is defined
    """
    if scope:
        scope[new_name] = cls
    cls.__qualname__ = cls.__qualname__[: -len(cls.__name__)] + new_name
    cls.__name__ = new_name


def unwrap_optional_type(type_: Any) -> Any:
    """Return the type of an Optional type, e.g. `unwrap_optional(Optional[str]) == str`;
    `unwrap_optional(Union[None, int, None]) == int`.

    :param type_: the type to unwrap
    :return: the "core" type of an Optional type
    """
    if not is_optional_type(type_):
        return type_
    for arg in get_args(type_):
        if arg is not type(None):
            return arg
