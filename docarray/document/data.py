import uuid
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .score import NamedScore
    from .. import DocumentArray
    from ..typing import ArrayType, StructValueType
    from datetime import datetime

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


@dataclass(unsafe_hash=True)
class DocumentData:
    id: str = field(default_factory=lambda: uuid.uuid1().hex)
    granularity: Optional[int] = None
    adjacency: Optional[int] = None
    parent_id: Optional[str] = None
    buffer: Optional[bytes] = None
    blob: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    text: Optional[str] = None
    weight: Optional[float] = None
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    tags: Optional[Dict[str, 'StructValueType']] = None
    offset: Optional[float] = None
    location: Optional[List[float]] = None
    embedding: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    modality: Optional[str] = None
    evaluations: Optional[Dict[str, 'NamedScore']] = None
    scores: Optional[Dict[str, 'NamedScore']] = None
    chunks: Optional['DocumentArray'] = None
    matches: Optional['DocumentArray'] = None
    timestamps: Optional[Dict[str, 'datetime']] = None

    def _clear_content(self):
        self.text = None
        self.blob = None
        self.buffer = None

    def __setattr__(self, key, value):
        if value is not None:
            if key == 'text' or key == 'blob' or key == 'buffer':
                # enable mutual exclusivity for content field
                if value != default_values.get(key):
                    self._clear_content()
            elif key == 'chunks' or key == 'matches':
                # force setter to DocumentArray
                from .. import DocumentArray
                value = DocumentArray(value)
        super().__setattr__(key, value)

    @property
    def non_empty_fields(self) -> Tuple[str]:
        """Get all non-emtpy fields of this :class:`Document`.

        Non-empty fields are the fields with not-`None` and not-default values.

        :return: field names in a tuple.
        """
        r = []
        for f in fields(self):
            f_name = f.name
            v = getattr(self, f_name)
            if v is not None:
                if f.name not in default_values:
                    r.append(f_name)
                elif v != default_values[f_name]:
                    r.append(f_name)
        return tuple(r)

    def _set_default_value_if_none(self, key):
        if getattr(self, key) is None:
            v = default_values.get(key, None)
            if v is not None:
                if v == 'DocumentArray':
                    from ... import DocumentArray

                    setattr(self, key, DocumentArray())
                else:
                    setattr(self, key, v() if callable(v) else v)
