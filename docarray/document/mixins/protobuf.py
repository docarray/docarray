from typing import TYPE_CHECKING, Type, Optional

if TYPE_CHECKING:
    from ...typing import T
    from ...proto.docarray_pb2 import DocumentProto


class ProtobufMixin:
    @classmethod
    def from_protobuf(cls: Type['T'], pb_msg: 'DocumentProto') -> 'T':
        from ...proto.io import parse_proto

        return parse_proto(pb_msg)

    def to_protobuf(self, ndarray_type: Optional[str] = None) -> 'DocumentProto':
        """Convert Document into a Protobuf message.

        :param ndarray_type: can be ``list`` or ``numpy``, if set it will force all ndarray-like object to be ``List`` or ``numpy.ndarray``.
        :return: the protobuf message
        """
        from ...proto.io import flush_proto

        return flush_proto(self, ndarray_type)
