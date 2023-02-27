import base64
import pickle
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from typing_inspect import is_union_type

from docarray.base_document.base_node import BaseNode
from docarray.typing.proto_register import _PROTO_TYPE_NAME_TO_CLASS
from docarray.utils.compress import _compress_bytes, _decompress_bytes

if TYPE_CHECKING:
    from pydantic.fields import ModelField

    from docarray.proto import DocumentProto, NodeProto


T = TypeVar('T', bound='IOMixin')


class IOMixin(Iterable[Tuple[str, Any]]):
    """
    IOMixin to define all the bytes/protobuf/json related part of BaseDocument
    """

    __fields__: Dict[str, 'ModelField']

    @classmethod
    @abstractmethod
    def _get_field_type(cls, field: str) -> Type:
        ...

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def to_bytes(
        self, protocol: str = 'protobuf', compress: Optional[str] = None
    ) -> bytes:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :return: the binary serialization in bytes
        """
        import pickle

        if protocol == 'pickle':
            bstr = pickle.dumps(self)
        elif protocol == 'protobuf':
            bstr = self.to_protobuf().SerializePartialToString()
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )
        return _compress_bytes(bstr, algorithm=compress)

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: binary bytes
        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a Document object
        """
        bstr = _decompress_bytes(data, algorithm=compress)
        if protocol == 'pickle':
            return pickle.loads(bstr)
        elif protocol == 'protobuf':
            from docarray.proto import DocumentProto

            pb_msg = DocumentProto()
            pb_msg.ParseFromString(bstr)
            return cls.from_protobuf(pb_msg)
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )

    def to_base64(
        self, protocol: str = 'protobuf', compress: Optional[str] = None
    ) -> str:
        """Serialize a Document object into as base64 string

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a base64 encoded string
        """
        return base64.b64encode(self.to_bytes(protocol, compress)).decode('utf-8')

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: a base64 encoded string
        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a Document object
        """
        return cls.from_bytes(base64.b64decode(data), protocol, compress)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentProto') -> T:
        """create a Document from a protobuf message

        :param pb_msg: the proto message of the Document
        :return: a Document initialize with the proto data
        """

        fields: Dict[str, Any] = {}

        for field_name in pb_msg.data:

            if field_name not in cls.__fields__.keys():
                continue  # optimization we don't even load the data if the key does not
                # match any field in the cls or in the mapping

            fields[field_name] = cls._get_content_from_node_proto(
                pb_msg.data[field_name], field_name
            )

        return cls(**fields)

    @classmethod
    def _get_content_from_node_proto(cls, value: 'NodeProto', field_name: str) -> Any:
        """
        load the proto data from a node proto

        :param value: the proto node value
        :param field_name: the name of the field
        :return: the loaded field
        """
        content_type_dict = _PROTO_TYPE_NAME_TO_CLASS
        arg_to_container: Dict[str, Callable] = {
            'list': list,
            'set': set,
            'tuple': tuple,
            'dict': dict,
        }

        content_key = value.WhichOneof('content')
        docarray_type = (
            value.type if value.WhichOneof('docarray_type') is not None else None
        )

        return_field: Any

        if docarray_type in content_type_dict:
            return_field = content_type_dict[docarray_type].from_protobuf(
                getattr(value, content_key)
            )
        elif content_key in ['document', 'document_array']:
            return_field = cls._get_field_type(field_name).from_protobuf(
                getattr(value, content_key)
            )  # we get to the parent class
        elif content_key is None:
            return_field = None
        elif docarray_type is None:

            if content_key in ['text', 'blob', 'integer', 'float', 'boolean']:
                return_field = getattr(value, content_key)

            elif content_key in arg_to_container.keys():
                from google.protobuf.json_format import MessageToDict

                return_field = arg_to_container[content_key](
                    MessageToDict(getattr(value, content_key))
                )

            else:
                raise ValueError(
                    f'key {content_key} is not supported for deserialization'
                )

        else:
            raise ValueError(
                f'type {docarray_type}, with key {content_key} is not supported for'
                f' deserialization'
            )

        return return_field

    def to_protobuf(self: T) -> 'DocumentProto':
        """Convert Document into a Protobuf message.

        :return: the protobuf message
        """
        from docarray.proto import DocumentProto, NodeProto

        data = {}
        for field, value in self:
            try:
                if isinstance(value, BaseNode):
                    nested_item = value._to_node_protobuf()

                elif isinstance(value, str):
                    nested_item = NodeProto(text=value)

                elif isinstance(value, bool):
                    nested_item = NodeProto(boolean=value)

                elif isinstance(value, int):
                    nested_item = NodeProto(integer=value)

                elif isinstance(value, float):
                    nested_item = NodeProto(float=value)

                elif isinstance(value, bytes):
                    nested_item = NodeProto(blob=value)

                elif isinstance(value, list):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(list=lvalue)

                elif isinstance(value, set):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(set=lvalue)

                elif isinstance(value, tuple):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(tuple=lvalue)

                elif isinstance(value, dict):
                    from google.protobuf.struct_pb2 import Struct

                    struct = Struct()
                    struct.update(value)
                    nested_item = NodeProto(dict=struct)
                elif value is None:
                    nested_item = NodeProto()
                else:
                    raise ValueError(f'field {field} with {value} is not supported')

                data[field] = nested_item

            except RecursionError as ex:
                if len(ex.args) >= 1:
                    ex.args = (
                        (
                            f'Field `{field}` contains cyclic reference in memory. '
                            'Could it be your Document is referring to itself?'
                        ),
                    )
                raise
            except Exception as ex:
                if len(ex.args) >= 1:
                    ex.args = (f'Field `{field}` is problematic',) + ex.args
                raise

        return DocumentProto(data=data)

    def _to_node_protobuf(self) -> 'NodeProto':
        from docarray.proto import NodeProto

        """Convert Document into a NodeProto protobuf message. This function should be
        called when the Document is nest into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(document=self.to_protobuf())

    @classmethod
    def _get_access_paths(cls) -> List[str]:
        """
        Get "__"-separated access paths of all fields, including nested ones.

        :return: list of all access paths
        """
        from docarray import BaseDocument

        paths = []
        for field in cls.__fields__.keys():
            field_type = cls._get_field_type(field)
            if not is_union_type(field_type) and issubclass(field_type, BaseDocument):
                sub_paths = field_type._get_access_paths()
                for path in sub_paths:
                    paths.append(f'{field}__{path}')
            else:
                paths.append(field)
        return paths
