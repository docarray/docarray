from typing import Any, get_args

from typing_inspect import is_union_type

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
