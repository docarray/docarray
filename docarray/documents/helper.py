from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, TypeVar

from pydantic import create_model, create_model_from_typeddict
from pydantic.config import BaseConfig
from typing_extensions import TypedDict
from docarray.utils._internal._typing import safe_issubclass

from docarray import BaseDoc

if TYPE_CHECKING:
    from pydantic.typing import AnyClassMethod

    T_doc = TypeVar('T_doc', bound=BaseDoc)


def create_doc(
    __model_name: str,
    *,
    __config__: Optional[Type[BaseConfig]] = None,
    __base__: Type['T_doc'] = BaseDoc,  # type: ignore
    __module__: str = __name__,
    __validators__: Dict[str, 'AnyClassMethod'] = None,  # type: ignore
    __cls_kwargs__: Dict[str, Any] = None,  # type: ignore
    __slots__: Optional[Tuple[str, ...]] = None,
    **field_definitions: Any,
) -> Type['T_doc']:
    """
    Dynamically create a subclass of BaseDoc. This is a wrapper around pydantic's create_model.

    ```python
    from docarray.documents import Audio
    from docarray.documents.helper import create_doc
    from docarray.typing.tensor.audio import AudioNdArray

    MyAudio = create_doc(
        'MyAudio',
        __base__=Audio,
        title=(str, ...),
        tensor=(AudioNdArray, ...),
    )

    assert safe_issubclass(MyAudio, BaseDoc)
    assert safe_issubclass(MyAudio, Audio)
    ```

    :param __model_name: name of the created model
    :param __config__: config class to use for the new model
    :param __base__: base class for the new model to inherit from, must be BaseDoc or its subclass
    :param __module__: module of the created model
    :param __validators__: a dict of method names and @validator class methods
    :param __cls_kwargs__: a dict for class creation
    :param __slots__: Deprecated, `__slots__` should not be passed to `create_model`
    :param field_definitions: fields of the model (or extra fields if a base is supplied)
        in the format `<name>=(<type>, <default default>)` or `<name>=<default value>`
    :return: the new Document class
    """

    if not safe_issubclass(__base__, BaseDoc):
        raise ValueError(f'{type(__base__)} is not a BaseDoc or its subclass')

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


def create_doc_from_typeddict(
    typeddict_cls: Type['TypedDict'],  # type: ignore
    **kwargs: Any,
):
    """
    Create a subclass of BaseDoc based on the fields of a `TypedDict`. This is a wrapper around pydantic's create_model_from_typeddict.

    ---

    ```python
    from typing_extensions import TypedDict

    from docarray import BaseDoc
    from docarray.documents import Audio
    from docarray.documents.helper import create_doc_from_typeddict
    from docarray.typing.tensor.audio import AudioNdArray


    class MyAudio(TypedDict):
        title: str
        tensor: AudioNdArray


    Doc = create_doc_from_typeddict(MyAudio, __base__=Audio)

    assert safe_issubclass(Doc, BaseDoc)
    assert safe_issubclass(Doc, Audio)
    ```

    ---

    :param typeddict_cls: TypedDict class to use for the new Document class
    :param kwargs: extra arguments to pass to `create_model_from_typeddict`
    :return: the new Document class
    """

    if '__base__' in kwargs:
        if not safe_issubclass(kwargs['__base__'], BaseDoc):
            raise ValueError(f'{kwargs["__base__"]} is not a BaseDoc or its subclass')
    else:
        kwargs['__base__'] = BaseDoc

    doc = create_model_from_typeddict(typeddict_cls, **kwargs)

    return doc


def create_doc_from_dict(model_name: str, data_dict: Dict[str, Any]) -> Type['T_doc']:
    """
    Create a subclass of BaseDoc based on example data given as a dictionary.

    In case the example contains None as a value,
    corresponding field will be viewed as the type Any.

    ---

    ```python
    import numpy as np
    from docarray.documents import ImageDoc
    from docarray.documents.helper import create_doc_from_dict

    data_dict = {'image': ImageDoc(tensor=np.random.rand(3, 224, 224)), 'author': 'me'}

    MyDoc = create_doc_from_dict(model_name='MyDoc', data_dict=data_dict)

    assert safe_issubclass(MyDoc, BaseDoc)
    ```

    ---

    :param model_name: Name of the new Document class
    :param data_dict: Dictionary of field types to their corresponding values.
    :return: the new Document class
    """
    if not data_dict:
        raise ValueError('`data_dict` should contain at least one item')

    field_types = {
        field: (type(value) if value else Any, ...)
        for field, value in data_dict.items()
    }
    return create_doc(__model_name=model_name, **field_types)  # type: ignore
