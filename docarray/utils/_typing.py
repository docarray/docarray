import collections
from types import GenericAlias
from typing import Any, _AnnotatedAlias, _GenericAlias

from typing_inspect import is_union_type

from docarray.typing.tensor.abstract_tensor import AbstractTensor


def get_args(tp):
    """Get type arguments with all substitutions performed.

    For unions, basic simplifications used by Union constructor are performed.
    Examples::
        get_args(Dict[str, int]) == (str, int)
        get_args(int) == ()
        get_args(Union[int, Union[T, int], str][int]) == (int, str)
        get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        get_args(Callable[[], T][int]) == ([], int)
    """
    if isinstance(tp, _AnnotatedAlias):
        return (tp.__origin__,) + tp.__metadata__
    if isinstance(tp, (_GenericAlias, GenericAlias)):
        res = tp.__args__
        if tp.__origin__ is collections.abc.Callable and res[0] is not Ellipsis:
            res = (list(res[:-1]), res[-1])
        return res
    return ()


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
