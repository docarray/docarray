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
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing import _GenericAlias as GenericAlias
from typing import get_origin

import numpy as np
from typing_inspect import get_args, is_union_type

from docarray.base_doc.base_node import BaseNode
from docarray.typing import NdArray
from docarray.typing.proto_register import _PROTO_TYPE_NAME_TO_CLASS
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.compress import _compress_bytes, _decompress_bytes
from docarray.utils._internal.misc import ProtocolType, import_library
from docarray.utils._internal.pydantic import is_pydantic_v2

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
    from pydantic.fields import FieldInfo

    from docarray.proto import DocProto, NodeProto
    from docarray.typing import TensorFlowTensor, TorchTensor

else:
    tf = import_library('tensorflow', raise_error=False)
    if tf is not None:
        from docarray.typing import TensorFlowTensor

    torch = import_library('torch', raise_error=False)
    if torch is not None:
        from docarray.typing import TorchTensor

T = TypeVar('T', bound='IOMixin')


def _type_to_protobuf(value: Any) -> 'NodeProto':
    """Convert any type to a NodeProto
    :param value: any object that need to be serialized
    :return: a NodeProto
    """
    from docarray.proto import NodeProto

    basic_type_to_key = {
        str: 'text',
        bool: 'boolean',
        int: 'integer',
        float: 'float',
        bytes: 'blob',
    }

    container_type_to_key = {list: 'list', set: 'set', tuple: 'tuple'}

    nested_item: 'NodeProto'

    if isinstance(value, BaseNode):
        nested_item = value._to_node_protobuf()
        return nested_item

    base_node_wrap: BaseNode
    if torch is not None:
        if isinstance(value, torch.Tensor):
            base_node_wrap = TorchTensor._docarray_from_native(value)
            return base_node_wrap._to_node_protobuf()

    if tf is not None:
        if isinstance(value, tf.Tensor):
            base_node_wrap = TensorFlowTensor._docarray_from_native(value)
            return base_node_wrap._to_node_protobuf()

    if isinstance(value, np.ndarray):
        base_node_wrap = NdArray._docarray_from_native(value)
        return base_node_wrap._to_node_protobuf()

    for basic_type, key_name in basic_type_to_key.items():
        if isinstance(value, basic_type):
            nested_item = NodeProto(**{key_name: value})
            return nested_item

    for container_type, key_name in container_type_to_key.items():
        if isinstance(value, container_type):
            from docarray.proto import ListOfAnyProto

            lvalue = ListOfAnyProto()
            for item in value:
                lvalue.data.append(_type_to_protobuf(item))
            nested_item = NodeProto(**{key_name: lvalue})
            return nested_item

    if isinstance(value, dict):
        from docarray.proto import DictOfAnyProto

        data = {}

        for key, content in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f'Protobuf only support string as key, but got {type(key)}'
                )

            data[key] = _type_to_protobuf(content)

        struct = DictOfAnyProto(data=data)
        nested_item = NodeProto(dict=struct)
        return nested_item

    elif value is None:
        nested_item = NodeProto()
        return nested_item
    else:
        raise ValueError(f'{type(value)} is not supported with protobuf')


class IOMixin(Iterable[Tuple[str, Any]]):
    """
    IOMixin to define all the bytes/protobuf/json related part of BaseDoc
    """

    _docarray_fields: Dict[str, 'FieldInfo']

    class Config:
        _load_extra_fields_from_protobuf: bool

    @classmethod
    @abstractmethod
    def _get_field_annotation(cls, field: str) -> Type:
        ...

    @classmethod
    def _get_field_annotation_array(cls, field: str) -> Type:
        return cls._get_field_annotation(field)

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def to_bytes(
        self, protocol: ProtocolType = 'protobuf', compress: Optional[str] = None
    ) -> bytes:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compression algorithm to use
        :return: the binary serialization in bytes
        """
        import pickle

        if protocol == 'pickle':
            bstr = pickle.dumps(self)
        elif protocol == 'protobuf':
            bstr = self.to_protobuf().SerializePartialToString()
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or '
                f'pickle protocols 0-5.'
            )
        return _compress_bytes(bstr, algorithm=compress)

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: ProtocolType = 'protobuf',
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
            from docarray.proto import DocProto

            pb_msg = DocProto()
            pb_msg.ParseFromString(bstr)
            return cls.from_protobuf(pb_msg)
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or '
                f'pickle protocols 0-5.'
            )

    def to_base64(
        self, protocol: ProtocolType = 'protobuf', compress: Optional[str] = None
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
        protocol: Literal['pickle', 'protobuf'] = 'pickle',
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
    def from_protobuf(cls: Type[T], pb_msg: 'DocProto') -> T:
        """create a Document from a protobuf message

        :param pb_msg: the proto message of the Document
        :return: a Document initialize with the proto data
        """

        fields: Dict[str, Any] = {}
        load_extra_field = (
            cls.model_config['_load_extra_fields_from_protobuf']
            if is_pydantic_v2
            else cls.Config._load_extra_fields_from_protobuf
        )
        for field_name in pb_msg.data:
            if (
                not (load_extra_field)
                and field_name not in cls._docarray_fields().keys()
            ):
                continue  # optimization we don't even load the data if the key does not
                # match any field in the cls or in the mapping

            fields[field_name] = cls._get_content_from_node_proto(
                pb_msg.data[field_name], field_name
            )

        return cls(**fields)

    @classmethod
    def _get_content_from_node_proto(
        cls,
        value: 'NodeProto',
        field_name: Optional[str] = None,
        field_type: Optional[Type] = None,
    ) -> Any:
        """
        load the proto data from a node proto

        :param value: the proto node value
        :param field_name: the name of the field
        :return: the loaded field
        """
        if field_name is not None and field_type is not None:
            raise ValueError("field_type and field_name cannot be both passed")

        field_type = field_type or (
            cls._get_field_annotation(field_name) if field_name else None
        )

        content_type_dict = _PROTO_TYPE_NAME_TO_CLASS

        content_key = value.WhichOneof('content')
        docarray_type = (
            value.type if value.WhichOneof('docarray_type') is not None else None
        )

        return_field: Any
        if docarray_type in content_type_dict:
            return_field = content_type_dict[docarray_type].from_protobuf(
                getattr(value, content_key)
            )
        elif content_key == 'doc':
            if field_type is None:
                raise ValueError(
                    'field_type cannot be None when trying to deserialize a BaseDoc'
                )
            try:
                return_field = field_type.from_protobuf(
                    getattr(value, content_key)
                )  # we get to the parent class
            except Exception:
                if get_origin(field_type) is Union:
                    raise ValueError(
                        'Union type is not supported for proto deserialization. Please use JSON serialization instead'
                    )
                raise ValueError(
                    f'{field_type} is not supported for proto deserialization'
                )
        elif content_key == 'doc_array':
            if field_type is not None and field_name is None:
                return_field = field_type.from_protobuf(getattr(value, content_key))
            elif field_name is not None:
                return_field = cls._get_field_annotation_array(
                    field_name
                ).from_protobuf(
                    getattr(value, content_key)
                )  # we get to the parent class
            else:
                raise ValueError(
                    'field_name and field_type cannot be None when trying to deserialize a DocArray'
                )
        elif content_key is None:
            return_field = None
        elif docarray_type is None:
            arg_to_container: Dict[str, Callable] = {
                'list': list,
                'set': set,
                'tuple': tuple,
            }

            if content_key in ['text', 'blob', 'integer', 'float', 'boolean']:
                return_field = getattr(value, content_key)

            elif content_key in arg_to_container.keys():
                if field_name and field_name in cls._docarray_fields():
                    field_type = cls._get_field_inner_type(field_name)

                if isinstance(field_type, GenericAlias):
                    field_type = get_args(field_type)[0]

                return_field = arg_to_container[content_key](
                    cls._get_content_from_node_proto(node, field_type=field_type)
                    for node in getattr(value, content_key).data
                )

            elif content_key == 'dict':
                deser_dict: Dict[str, Any] = dict()

                if field_name and field_name in cls._docarray_fields():
                    if is_pydantic_v2:
                        dict_args = get_args(
                            cls._docarray_fields()[field_name].annotation
                        )
                        if len(dict_args) < 2:
                            field_type = Any
                        else:
                            field_type = dict_args[1]
                    else:
                        field_type = cls._docarray_fields()[field_name].type_

                else:
                    field_type = None

                for key_name, node in value.dict.data.items():
                    deser_dict[key_name] = cls._get_content_from_node_proto(
                        node, field_type=field_type
                    )
                return_field = deser_dict
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

    def to_protobuf(self: T) -> 'DocProto':
        """Convert Document into a Protobuf message.

        :return: the protobuf message
        """
        from docarray.proto import DocProto

        data = {}
        for field, value in self:
            try:
                data[field] = _type_to_protobuf(value)
            except RecursionError as ex:
                if len(ex.args) >= 1:
                    ex.args = (
                        (
                            f'Field `{field}` contains cyclic reference in memory. '
                            'Could it be your Document is referring to itself?'
                        ),
                    )
                raise ex
            except Exception as ex:
                if len(ex.args) >= 1:
                    ex.args = (f'Field `{field}` is problematic',) + ex.args
                raise ex

        return DocProto(data=data)

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should be
        called when the Document is nest into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(doc=self.to_protobuf())

    @classmethod
    def _get_access_paths(cls) -> List[str]:
        """
        Get "__"-separated access paths of all fields, including nested ones.

        :return: list of all access paths
        """
        from docarray import BaseDoc

        paths = []
        for field in cls._docarray_fields().keys():
            field_type = cls._get_field_annotation(field)
            if not is_union_type(field_type) and safe_issubclass(field_type, BaseDoc):
                sub_paths = field_type._get_access_paths()
                for path in sub_paths:
                    paths.append(f'{field}__{path}')
            else:
                paths.append(field)
        return paths

    @classmethod
    def from_json(
        cls: Type[T],
        data: str,
    ) -> T:
        """Build Document object from json data
        :return: a Document object
        """
        # TODO: add tests

        if is_pydantic_v2:
            return cls.model_validate_json(data)
        else:
            return cls.parse_raw(data)
