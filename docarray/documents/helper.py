from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, TypeVar

from pydantic import create_model, create_model_from_typeddict
from pydantic.config import BaseConfig
from typing_extensions import TypedDict

from docarray.base_document import BaseDocument

if TYPE_CHECKING:
    from pydantic.typing import AnyClassMethod

    T_doc = TypeVar('T_doc', bound=BaseDocument)


def create_doc(
    __model_name: str,
    *,
    __config__: Optional[Type[BaseConfig]] = None,
    __base__: Type['T_doc'] = BaseDocument,  # type: ignore
    __module__: str = __name__,
    __validators__: Dict[str, 'AnyClassMethod'] = None,  # type: ignore
    __cls_kwargs__: Dict[str, Any] = None,  # type: ignore
    __slots__: Optional[Tuple[str, ...]] = None,
    **field_definitions: Any,
) -> Type['T_doc']:
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
    :return: the new Document class

    EXAMPLE USAGE

    .. code-block:: python

        from docarray.documents import Audio
        from docarray.documents.helper import create_doc
        from docarray.typing.tensor.audio import AudioNdArray

        MyAudio = create_doc(
            'MyAudio',
            __base__=Audio,
            title=(str, ...),
            tensor=(AudioNdArray, ...),
        )

        assert issubclass(MyAudio, BaseDocument)
        assert issubclass(MyAudio, Audio)

    """

    if not issubclass(__base__, BaseDocument):
        raise ValueError(f'{type(__base__)} is not a BaseDocument or its subclass')

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
):
    """
    Create a subclass of BaseDocument based on the fields of a `TypedDict`. This is a wrapper around pydantic's create_model_from_typeddict.
    :param typeddict_cls: TypedDict class to use for the new Document class
    :param kwargs: extra arguments to pass to `create_model_from_typeddict`
    :return: the new Document class

    EXAMPLE USAGE

    .. code-block:: python

        from typing_extensions import TypedDict

        from docarray import BaseDocument
        from docarray.documents import Audio
        from docarray.documents.helper import create_from_typeddict
        from docarray.typing.tensor.audio import AudioNdArray


        class MyAudio(TypedDict):
            title: str
            tensor: AudioNdArray


        Doc = create_from_typeddict(MyAudio, __base__=Audio)

        assert issubclass(Doc, BaseDocument)
        assert issubclass(Doc, Audio)

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

def create_from_dict(
    data: Dict[str,Any],  # type: ignore
    **kwargs: Any,
):
    if '__base__' in kwargs:
        if not issubclass(kwargs['__base__'], BaseDocument):
            raise ValueError(
                f'{kwargs["__base__"]} is not a BaseDocument or its subclass'
            )
    else:
        kwargs['__base__'] = BaseDocument
    typed_dict_fields = {}
    for key, value in data.items():
        field_type = type(value)
        typed_dict_fields[key] = field_type
        print(typed_dict_fields[key])
    
    GeneratedTypedDict = TypedDict('GeneratedTypedDict', typed_dict_fields)
    doc = create_model_from_typeddict(GeneratedTypedDict, **kwargs)
    return doc