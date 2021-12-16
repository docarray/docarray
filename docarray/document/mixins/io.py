from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ...types import T
    from ...proto.docarray_pb2 import DocumentProto


class IOMixin:

    def to_protobuf(self) -> 'DocumentProto':
        if not hasattr(self, '_pb_body'):
            from ...proto.docarray_pb2 import DocumentProto

            self._pb_body = DocumentProto()
        self._pb_body.Clear()
        from ...proto.io import flush_proto

        # only flush those non-empty fields to Protobuf
        for k in self._data.non_empty_fields:
            v = getattr(self, k)
            flush_proto(self._pb_body, k, v)
        return self._pb_body

    @classmethod
    def from_protobuf(cls: Type['T'], pb_msg: 'DocumentProto') -> 'T':
        ...


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
