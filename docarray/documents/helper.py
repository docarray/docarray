from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, TypeVar, Union

from pydantic import create_model, create_model_from_typeddict
from pydantic.config import BaseConfig
from typing_extensions import TypedDict

from docarray import BaseDocument

if TYPE_CHECKING:
    from pydantic.typing import AnyClassMethod

    Document = TypeVar('Document', bound='BaseDocument')


def create_doc(
    __model_name: str,
    *,
    __config__: Optional[Type[BaseConfig]] = None,
    __base__: Union[
        None, Type['Document'], Tuple[Type['Document'], ...]
    ] = BaseDocument,
    __module__: str = __name__,
    __validators__: Dict[str, 'AnyClassMethod'] = None,
    __cls_kwargs__: Dict[str, Any] = None,
    __slots__: Optional[Tuple[str, ...]] = None,
    **field_definitions: Any,
) -> Type['BaseDocument']:
    """
    Dynamically create a subclass of BaseDocument. This is a wrapper around pydantic's create_model.
    :param __model_name: name of the created model
    :param __config__: config class to use for the new model
    :param __base__: base class for the new model to inherit from, must be BaseDocument or its subclass
    :param __module__: module of the created model
    :param __validators__: a dict of method names and @validator class methods
    :param __cls_kwargs__: a dict for class creation
    :param __slots__: Deprecated, `__slots__` should not be passed to `create_model`
    :param field_definitions: fields of the model (or extra fields if a base is supplied)
        in the format `<name>=(<type>, <default default>)` or `<name>=<default value>`
    """

    if not issubclass(__base__, BaseDocument):
        raise ValueError(f'{__base__} is not a BaseDocument or its subclass')

    doc = create_model(
        __model_name,
        __config__=__config__,
        __base__=__base__,
        __module__=__module__,
        __validators__=__validators__,
        __cls_kwargs__=__cls_kwargs__,
        __slots__=__slots__,
        **field_definitions,
    )

    return doc


def create_from_typeddict(
    typeddict_cls: Type['TypedDict'],  # type: ignore
    **kwargs: Any,
) -> Type[BaseDocument]:
    """
    Create a subclass of BaseDocument based on the fields of a `TypedDict`. This is a wrapper around pydantic's create_model_from_typeddict.
    """

    if '__base__' in kwargs:
        if not issubclass(kwargs['__base__'], BaseDocument):
            raise ValueError(
                f'{kwargs["__base__"]} is not a BaseDocument or its subclass'
            )
    else:
        kwargs['__base__'] = BaseDocument

    doc = create_model_from_typeddict(typeddict_cls, **kwargs)

    return doc
