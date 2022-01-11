import dataclasses
import pickle
from typing import Optional, TYPE_CHECKING, Type, Dict, Any
import base64

from ...helper import compress_bytes, decompress_bytes

if TYPE_CHECKING:
    from ...types import T


class PortingMixin:
    @classmethod
    def from_dict(cls: Type['T'], obj: Dict) -> 'T':
        from google.protobuf import json_format
        from ...proto.docarray_pb2 import DocumentProto

        pb_msg = DocumentProto()
        json_format.ParseDict(obj, pb_msg)
        return cls.from_protobuf(pb_msg)

    @classmethod
    def from_json(cls: Type['T'], obj: str) -> 'T':
        from google.protobuf import json_format
        from ...proto.docarray_pb2 import DocumentProto

        pb_msg = DocumentProto()
        json_format.Parse(obj, pb_msg)
        return cls.from_protobuf(pb_msg)

    def to_dict(self, strict: bool = True) -> Dict[str, Any]:
        if strict:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(
                self.to_protobuf(),
                preserving_proto_field_name=True,
            )
        else:
            return dataclasses.asdict(self._data)

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
            d = pickle.loads(bstr)
        elif protocol == 'protobuf':
            from ...proto.docarray_pb2 import DocumentProto

            pb_msg = DocumentProto()
            pb_msg.ParseFromString(bstr)
            d = cls.from_protobuf(pb_msg)
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )
        return d

    def to_json(self) -> str:
        from google.protobuf.json_format import MessageToJson

        return MessageToJson(
            self.to_protobuf(), preserving_proto_field_name=True, sort_keys=True
        )

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
