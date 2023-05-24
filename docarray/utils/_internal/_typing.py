from typing import Any, Optional

from typing_inspect import get_args, is_union_type

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


def is_tensor_any(first_doc, field_name, field_type, tensor_type) -> bool:
    """Return True if the `field_name` in the doc is a subclass of the tensor type."""
    # all generic tensor types such as AnyTensor, AnyEmbedding, ImageTensor, etc. are subclasses of AbstractTensor
    if issubclass(field_type, AbstractTensor):
        # check if the type of the field_name in doc is a subclass of the tensor type
        # e.g. if the field_type is AnyTensor but the tensor type is ImageTensor, then we return True so that
        # specific type is picked and not the generic type
        tensor = getattr(first_doc, field_name)
        if issubclass(tensor.__class__, tensor_type):
            return True
    return False


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
