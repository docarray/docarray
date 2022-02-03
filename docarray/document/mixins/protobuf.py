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
        from ...proto.io import flush_proto

        return flush_proto(self)
