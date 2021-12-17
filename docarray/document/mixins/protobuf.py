from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ...types import T
    from ...proto.docarray_pb2 import DocumentProto


class ProtobufMixin:

    @classmethod
    def from_protobuf(cls: Type['T'], pb_msg: 'DocumentProto') -> 'T':
        from ...proto.io import parse_proto

        return parse_proto(pb_msg)

    def to_protobuf(self) -> 'DocumentProto':
        from ...proto.docarray_pb2 import DocumentProto
        from ...proto.io import flush_proto
        pb_msg = DocumentProto()

        # only flush those non-empty fields to Protobuf
        for k in self.non_empty_fields:
            flush_proto(pb_msg, k, getattr(self, k))
        return pb_msg

    def to_dict(self):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(
            self.to_protobuf(),
            preserving_proto_field_name=True,
        )

    def to_bytes(self) -> bytes:
        return self.to_protobuf().SerializePartialToString()

    def to_json(self):
        from google.protobuf.json_format import MessageToJson

        return MessageToJson(
            self.to_protobuf(), preserving_proto_field_name=True, sort_keys=True
        )
