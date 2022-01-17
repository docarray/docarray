import mimetypes
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..score import NamedScore
    from .. import DocumentArray, Document
    from ..types import ArrayType, StructValueType, DocumentContentType

default_values = dict(
    granularity=0,
    adjacency=0,
    parent_id='',
    blob=b'',
    text='',
    weight=0.0,
    uri='',
    mime_type='',
    tags=dict,
    offset=0.0,
    location=list,
    modality='',
    evaluations='Dict[str, NamedScore]',
    scores='Dict[str, NamedScore]',
    chunks='ChunkArray',
    matches='MatchArray',
    timestamps=dict,
)

_all_mime_types = set(mimetypes.types_map.values())


@dataclass(unsafe_hash=True)
class DocumentData:
    _reference_doc: 'Document' = field(hash=False, compare=False)
    id: str = field(default_factory=lambda: uuid.uuid1().hex)
    parent_id: Optional[str] = None
    granularity: Optional[int] = None
    adjacency: Optional[int] = None
    blob: Optional[bytes] = None
    tensor: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    mime_type: Optional[str] = None  # must be put in front of `text` `content`
    text: Optional[str] = None
    content: Optional['DocumentContentType'] = None
    weight: Optional[float] = None
    uri: Optional[str] = None
    tags: Optional[Dict[str, 'StructValueType']] = None
    offset: Optional[float] = None
    location: Optional[List[float]] = None
    embedding: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    modality: Optional[str] = None
    evaluations: Optional[Dict[str, Union['NamedScore', Dict]]] = None
    scores: Optional[Dict[str, Union['NamedScore', Dict]]] = None
    chunks: Optional['DocumentArray'] = None
    matches: Optional['DocumentArray'] = None

    def __setattr__(self, key, value):
        if value is not None:
            if key == 'text' or key == 'tensor' or key == 'blob':
                # enable mutual exclusivity for content field
                dv = default_values.get(key)
                if type(value) != type(dv) or value != dv:
                    self.text = None
                    self.tensor = None
                    self.blob = None
                    if key == 'text':
                        self.mime_type = 'text/plain'
            elif key == 'uri':
                mime_type = mimetypes.guess_type(value)[0]

                if mime_type:
                    self.mime_type = mime_type
            elif key == 'mime_type':
                if value not in _all_mime_types:
                    # given but not recognizable, do best guess
                    r = mimetypes.guess_type(f'*.{value}')[0]
                    value = r or value
            elif key == 'content':
                if isinstance(value, bytes):
                    self.blob = value
                elif isinstance(value, str):
                    self.text = value
                else:
                    self.tensor = value
                value = None
            elif key == 'chunks':
                from ..array.chunk import ChunkArray

                if not isinstance(value, ChunkArray):
                    value = ChunkArray(value, reference_doc=self._reference_doc)
            elif key == 'matches':
                from ..array.match import MatchArray

                if not isinstance(value, MatchArray):
                    value = MatchArray(value, reference_doc=self._reference_doc)
        self.__dict__[key] = value

    @property
    def _non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self):
            f_name = f.name
            if not f_name.startswith('_'):
                v = getattr(self, f_name)
                if v is not None:
                    if f_name not in default_values:
                        r.append(f_name)
                    else:
                        dv = default_values[f_name]
                        if dv in (
                            'ChunkArray',
                            'MatchArray',
                            'DocumentArray',
                            list,
                            dict,
                            'Dict[str, NamedScore]',
                        ):
                            if v:
                                r.append(f_name)
                        elif v != dv:
                            r.append(f_name)

        return tuple(r)

    def _set_default_value_if_none(self, key):
        if getattr(self, key) is None:
            v = default_values.get(key, None)
            if v is not None:
                if v == 'DocumentArray':
                    from .. import DocumentArray

                    setattr(self, key, DocumentArray())
                elif v == 'ChunkArray':
                    from ..array.chunk import ChunkArray

                    setattr(
                        self, key, ChunkArray(None, reference_doc=self._reference_doc)
                    )
                elif v == 'MatchArray':
                    from ..array.match import MatchArray

                    setattr(
                        self, key, MatchArray(None, reference_doc=self._reference_doc)
                    )
                elif v == 'Dict[str, NamedScore]':
                    from ..score import NamedScore

                    setattr(self, key, defaultdict(NamedScore))
                else:
                    setattr(self, key, v() if callable(v) else v)
