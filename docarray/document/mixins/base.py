import copy as cp
from typing import TYPE_CHECKING

from ...helper import typename

if TYPE_CHECKING:
    from ...typing import T
    from google.protobuf.message import Message


class BaseDocumentMixin:

    def copy_from(self: 'T', other: 'T') -> None:
        """Overwrite self by copying from another :class:`Document`.

        :param other: the other Document to copy from
        """
        self._data = cp.deepcopy(other._data)

    def clear(self) -> None:
        """Clear all fields from this :class:`Document` to their default values."""
        for f in self._data.non_empty_fields:
            setattr(self._data, f, None)

    def pop(self, *fields) -> None:
        """Clear some fields from this :class:`Document` to their default values.

        :param fields: field names to clear.
        """
        for f in fields:
            if hasattr(self, f):
                setattr(self._data, f, None)

    def to_dict(self):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(
            self.to_protobuf(),
            preserving_proto_field_name=True,
        )

    def to_protobuf(self) -> 'Message':
        if not hasattr(self, '_pb_body'):
            from ...proto.docarray_pb2 import DocumentProto
            self._pb_body = DocumentProto()
        self._pb_body.Clear()
        from ...proto.flush import flush_proto
        # only flush those non-empty fields to Protobuf
        for k in self._data.non_empty_fields:
            v = getattr(self, k)
            flush_proto(self._pb_body, k, v)
        return self._pb_body

    def to_bytes(self) -> bytes:
        return self.to_protobuf().SerializePartialToString()

    def to_json(self):
        from google.protobuf.json_format import MessageToJson

        return MessageToJson(
            self.to_protobuf(), preserving_proto_field_name=True, sort_keys=True
        )

    @property
    def nbytes(self) -> int:
        """Return total bytes consumed by protobuf.

        :return: number of bytes
        """
        return len(bytes(self))

    def __hash__(self):
        return hash(self._data)

    def __repr__(self):
        content = str(self._data.non_empty_fields)
        content += f' at {id(self)}'
        return f'<{typename(self)} {content.strip()}>'

    def __bytes__(self):
        return self.to_bytes()

    def __eq__(self, other):
        return self._data == other._data
