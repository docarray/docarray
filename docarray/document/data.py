import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .score import NamedScore
    from .. import DocumentArray
    from ..typing import ArrayType, DocumentContentType, StructValueType
    from datetime import datetime


@dataclass
class DocumentData:
    id: str = field(default_factory=lambda: uuid.uuid1().hex)
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
    timestamps: Optional[Dict[str, 'datetime']] = None
