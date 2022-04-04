import base64
import json
import pickle
from typing import Optional, TYPE_CHECKING, Type, Dict, Any, Union

from ...helper import compress_bytes, decompress_bytes

if TYPE_CHECKING:
    from ...typing import T


class PortingMixin:
    @classmethod
    def from_dict(
        cls: Type['T'], obj: Dict, protocol: str = 'jsonschema', **kwargs
    ) -> 'T':
        """Convert a dict object into a Document.

        :param obj: a Python dict object
        :param protocol: `jsonschema` or `protobuf`
        :param kwargs: extra key-value args pass to pydantic and protobuf parser.
        :return: the parsed Document object
        """
        if protocol == 'jsonschema':
            from ..pydantic_model import PydanticDocument

            return cls.from_pydantic_model(PydanticDocument.parse_obj(obj, **kwargs))
        elif protocol == 'protobuf':
            from google.protobuf import json_format
            from ...proto.docarray_pb2 import DocumentProto

            pb_msg = DocumentProto()
            json_format.ParseDict(obj, pb_msg, **kwargs)
            return cls.from_protobuf(pb_msg)
        else:
            return cls(obj)

    @classmethod
    def from_json(
        cls: Type['T'],
        obj: Union[str, bytes, bytearray],
        protocol: str = 'jsonschema',
        **kwargs,
    ) -> 'T':
        """Convert a JSON string into a Document.

        :param obj: a valid JSON string
        :param protocol: `jsonschema` or `protobuf`
        :param kwargs: extra key-value args pass to pydantic and protobuf parser.
        :return: the parsed Document object
        """
        if protocol == 'jsonschema':
            from ..pydantic_model import PydanticDocument

            return cls.from_pydantic_model(PydanticDocument.parse_raw(obj, **kwargs))
        elif protocol == 'protobuf':
            from google.protobuf import json_format
            from ...proto.docarray_pb2 import DocumentProto

            pb_msg = DocumentProto()
            json_format.Parse(obj, pb_msg, **kwargs)
            return cls.from_protobuf(pb_msg)
        else:
            return cls.from_dict(json.loads(obj), protocol=protocol)

    def to_dict(self, protocol: str = 'jsonschema', **kwargs) -> Dict[str, Any]:
        """Convert itself into a Python dict object.

        :param protocol: `jsonschema` or `protobuf`
        :param kwargs: extra key-value args pass to pydantic and protobuf dumper.
        :return: the dumped Document as a dict object
        """
        if protocol == 'jsonschema':
            return self.to_pydantic_model().dict(**kwargs)
        elif protocol == 'protobuf':
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(
                self.to_protobuf(),
                **kwargs,
            )
        else:
            raise ValueError(f'protocol=`{protocol}` is not supported')

    def to_bytes(
        self, protocol: str = 'pickle', compress: Optional[str] = None
    ) -> bytes:
        if protocol == 'pickle':
            bstr = pickle.dumps(self)
        elif protocol == 'protobuf':
            bstr = self.to_protobuf().SerializePartialToString()
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )
        return compress_bytes(bstr, algorithm=compress)

    @classmethod
    def from_bytes(
        cls: Type['T'],
        data: bytes,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> 'T':
        """Build Document object from binary bytes

        :param data: binary bytes
        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a Document object
        """
        bstr = decompress_bytes(data, algorithm=compress)
        if protocol == 'pickle':
            return pickle.loads(bstr)
        elif protocol == 'protobuf':
            from ...proto.docarray_pb2 import DocumentProto

            pb_msg = DocumentProto()
            pb_msg.ParseFromString(bstr)
            return cls.from_protobuf(pb_msg)
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )

    def to_json(self, protocol: str = 'jsonschema', **kwargs) -> str:
        """Convert itself into a JSON string.

        :param protocol: `jsonschema` or `protobuf`
        :param kwargs: extra key-value args pass to pydantic and protobuf dumper.
        :return: the dumped JSON string
        """
        if protocol == 'jsonschema':
            return self.to_pydantic_model().json(**kwargs)
        elif protocol == 'protobuf':
            from google.protobuf.json_format import MessageToJson

            return MessageToJson(self.to_protobuf(), **kwargs)
        else:
            raise ValueError(f'protocol={protocol} is not supported.')

    def to_base64(
        self, protocol: str = 'pickle', compress: Optional[str] = None
    ) -> str:
        """Serialize a Document object into as base64 string

        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a base64 encoded string
        """
        return base64.b64encode(self.to_bytes(protocol, compress)).decode('utf-8')

    @classmethod
    def from_base64(
        cls: Type['T'],
        data: str,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> 'T':
        """Build Document object from binary bytes

        :param data: a base64 encoded string
        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a Document object
        """
        return cls.from_bytes(base64.b64decode(data), protocol, compress)

    def _to_stream_bytes(self, protocol, compress) -> bytes:
        # 4 bytes (uint32)
        doc_as_bytes = self.to_bytes(protocol=protocol, compress=compress)

        # variable size bytes
        len_doc_as_bytes = len(doc_as_bytes).to_bytes(4, 'big', signed=False)

        return len_doc_as_bytes + doc_as_bytes
