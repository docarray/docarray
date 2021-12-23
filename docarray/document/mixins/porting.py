import pickle
from typing import Optional, TYPE_CHECKING, Type, Dict

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

    def to_dict(self):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(
            self.to_protobuf(),
            preserving_proto_field_name=True,
        )

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
