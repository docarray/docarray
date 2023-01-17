from typing import Callable
from docarray.typing.abstract_type import AbstractType
from typing import Type


_PROTO_TYPE_NAME_TO_CLASS = {}


def register_proto(
    proto_type_name: str,
) -> Callable[[Type[AbstractType]], Type[AbstractType]]:
    """Register a new type to be used in the protobuf serialization.
    :param cls: the class to register
    :return: the class
    """
    def _register(cls: Type['AbstractType']) -> Type['AbstractType']:
        cls._proto_type_name = proto_type_name

        _PROTO_TYPE_NAME_TO_CLASS[proto_type_name] = cls
        return cls

    return _register
