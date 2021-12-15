import copy as cp
from dataclasses import fields
from typing import TYPE_CHECKING, Dict, Tuple, Any

from ...helper import typename, cached_property

if TYPE_CHECKING:
    from ...typing import T
    from google.protobuf.message import Message

default_values = dict(
    granularity=0,
    adjacency=0,
    parent_id='',
    buffer=b'',
    text='',
    weight=0.0,
    uri='',
    mime_type='',
    tags=dict,
    offset=0.0,
    location=list,
    modality='',
    evaluations=list,
    scores=dict,
    chunks='DocumentArray',
    matches='DocumentArray',
    timestamps=dict,
)


class BaseDocumentMixin:
    def __eq__(self, other):
        return self._doc_data == self._doc_data

    def _set_default_value_if_none(self, key):
        if getattr(self._doc_data, key) is None:
            v = default_values.get(key, None)
            if v is not None:
                if v == 'DocumentArray':
                    from ... import DocumentArray

                    setattr(self._doc_data, key, DocumentArray())
                else:
                    setattr(self._doc_data, key, v() if callable(v) else v)

    @property
    def non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self._doc_data):
            f_name = f.name
            v = getattr(self._doc_data, f_name)
            if v is not None:
                if f.name not in default_values:
                    r.append(f_name)
                elif v != default_values[f_name]:
                    r.append(f_name)
        return tuple(r)

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
