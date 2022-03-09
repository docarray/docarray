import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from ..score import NamedScore
    from .. import DocumentArray, Document
    from ..types import ArrayType, StructValueType, DocumentContentType


@dataclass(unsafe_hash=True)
class DocumentData:
    _reference_doc: 'Document' = field(hash=False, compare=False)
    id: str = field(default_factory=lambda: os.urandom(16).hex())
    parent_id: str = ''
    granularity: int = 0
    adjacency: int = 0
    blob: bytes = b''
    tensor: 'ArrayType' = field(default=None, hash=False, compare=False)
    mime_type: str = ''  # must be put in front of `text` `content`
    text: str = ''
    content: 'DocumentContentType' = None
    weight: float = 0.0
    uri: str = ''
    tags: Dict[str, 'StructValueType'] = field(default_factory=dict)
    offset: float = 0.0
    location: List[float] = field(default_factory=list)
    embedding: 'ArrayType' = field(default=None, hash=False, compare=False)
    modality: str = ''
    evaluations: Dict[str, Union['NamedScore', Dict]] = None
    scores: Dict[str, Union['NamedScore', Dict]] = None
    chunks: 'DocumentArray' = None
    matches: 'DocumentArray' = None

    @property
    def _non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self):
            f_name = f.name
            if not f_name.startswith('_'):
                v = getattr(self, f_name)
                if v is not None:
                    if f_name in ('embedding', 'tensor'):
                        r.append(f_name)
                    elif v:
                        r.append(f_name)

        return tuple(r)
