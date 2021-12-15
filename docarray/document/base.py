import copy as cp
from dataclasses import dataclass
from dataclasses import fields
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

from ..helper import typename, cached_property

if TYPE_CHECKING:
    from .score import NamedScore
    from .. import DocumentArray
    from ..typing import ArrayType, DocumentContentType, StructValueType, T
    from google.protobuf.message import Message


@dataclass
class DocumentData:
    id: Optional[str] = None
    granularity: Optional[int] = None
    adjacency: Optional[int] = None
    parent_id: Optional[str] = None
    buffer: Optional[bytes] = None
    blob: Optional['ArrayType'] = None
    text: Optional[str] = None
    content: Optional['DocumentContentType'] = None
    weight: Optional[float] = None
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    tags: Optional[Dict[str, 'StructValueType']] = None
    offset: Optional[float] = None
    location: Optional[List[float]] = None
    embedding: Optional['ArrayType'] = None
    modality: Optional[str] = None
    evaluations: Optional[Dict[str, 'NamedScore']] = None
    scores: Optional[Dict[str, 'NamedScore']] = None
    chunks: Optional['DocumentArray'] = None
    matches: Optional['DocumentArray'] = None
    timestamps: Optional[Dict[str, datetime]] = None




class BaseDocument:

    def __eq__(self, other):
        return self._doc_data == self._doc_data

    @property
    def non_empty_fields(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self._doc_data) if getattr(self, f.name) is not None)

    def __copy__(self):
        return type(self)(self)

    def __deepcopy__(self, memodict={}):
        return type(self)(self, copy=True)

    def __repr__(self):
        content = str(self.non_empty_fields)
        content += f' at {id(self)}'
        return f'<{typename(self)} {content.strip()}>'

    def copy_from(self: 'T', other: 'T') -> None:
        """Copy the content of target

        :param other: the document to copy from
        """
        self._doc_data = cp.deepcopy(other._doc_data)

    @cached_property
    def _default_values(self) -> Dict[str, Any]:
        return {f.name: f.default for f in fields(self._doc_data)}

    def clear(self) -> None:
        for f in self.non_empty_fields:
            setattr(self._doc_data, f, self._default_values[f])

    def pop(self, *fields) -> None:
        for f in fields:
            setattr(self._doc_data, f, self._default_values[f])

    def to_dict(self):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(
            self.to_protobuf(),
            preserving_proto_field_name=True,
        )

    def to_protobuf(self) -> 'Message':
        ...

    def to_bytes(self) -> bytes:
        return self.to_protobuf().SerializePartialToString()

    def to_json(self):
        from google.protobuf.json_format import MessageToJson

        return MessageToJson(
            self.to_protobuf(), preserving_proto_field_name=True, sort_keys=True
        )

    def __bytes__(self):
        return self.to_bytes()

    @property
    def nbytes(self) -> int:
        """Return total bytes consumed by protobuf.

        :return: number of bytes
        """
        return len(bytes(self))

    def __hash__(self):
        return hash(self._doc_data)
