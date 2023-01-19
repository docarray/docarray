from typing import Callable, Type

from docarray.typing.abstract_type import AbstractType

_PROTO_TYPE_NAME_TO_CLASS = {}


def register_proto(
    proto_type_name: str,
) -> Callable[[Type[AbstractType]], Type[AbstractType]]:
    """Register a new type to be used in the protobuf serialization.

    This will add the type key to the global registry of types key used in the proto
    serialization and deserialization. This is for internal usage only.

     EXAMPLE USAGE

        .. code-block:: python

            from docarray.typing.proto_register import register_proto
            from docarray.typing.abstract_type import AbstractType


            @register_proto(proto_type_name='my_type')
            class MyType(AbstractType):
                ...

    :param cls: the class to register
    :return: the class
    """

    def _register(cls: Type['AbstractType']) -> Type['AbstractType']:
        cls._proto_type_name = proto_type_name

        _PROTO_TYPE_NAME_TO_CLASS[proto_type_name] = cls
        return cls

    return _register
