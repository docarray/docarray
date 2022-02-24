import base64
import json
from typing import List, Dict, Union, NewType, Any, Optional

import numpy as np
import strawberry

from ..math.ndarray import to_list

_ProtoValueType = Union[bool, float, str]

_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]

JSONScalar = strawberry.scalar(
    NewType('JSONScalar', Any),
    serialize=lambda v: v,
    parse_value=lambda v: json.loads(v),
    description="The GenericScalar scalar type represents a generic GraphQL scalar value that could be: List or Object.",
)

Base64 = strawberry.scalar(
    NewType('Base64', bytes),
    serialize=lambda v: base64.b64encode(v).decode('utf-8'),
    parse_value=lambda v: base64.b64decode(v.encode('utf-8')),
)

NdArray = strawberry.scalar(
    NewType('NdArray', bytes),
    serialize=lambda v: to_list(v),
    parse_value=lambda v: np.array(v),
)


### interface


@strawberry.interface
class _NamedScoreInterface:
    value: Optional[float] = None
    op_name: Optional[str] = None
    description: Optional[str] = None
    ref_id: Optional[str] = None


@strawberry.interface
class _BaseStrawberryDocumentInterface:
    id: Optional[str] = None
    parent_id: Optional[str] = None
    granularity: Optional[int] = None
    adjacency: Optional[int] = None
    blob: Optional[Base64] = None
    tensor: Optional[NdArray] = None
    mime_type: Optional[str] = None
    text: Optional[str] = None
    weight: Optional[float] = None
    uri: Optional[str] = None
    tags: Optional[JSONScalar] = None
    offset: Optional[float] = None
    location: Optional[List[float]] = None
    embedding: Optional[NdArray] = None
    modality: Optional[str] = None


### type


@strawberry.type
class _NamedScore(_NamedScoreInterface):
    ...


@strawberry.type
class _NameScoreItem:
    name: str
    score: _NamedScore


@strawberry.type
class StrawberryDocument(strawberry.type(_BaseStrawberryDocumentInterface)):
    evaluations: Optional[List[_NameScoreItem]] = None
    scores: Optional[List[_NameScoreItem]] = None
    chunks: Optional[List['StrawberryDocument']] = None
    matches: Optional[List['StrawberryDocument']] = None


### input


@strawberry.input
class _NamedScoreInput(_NamedScoreInterface):
    ...


@strawberry.input
class _NameScoreItemInput:
    name: str
    score: _NamedScoreInput


@strawberry.input
class StrawberryDocumentInput(strawberry.input(_BaseStrawberryDocumentInterface)):
    evaluations: Optional[List[_NameScoreItemInput]] = None
    scores: Optional[List[_NameScoreItemInput]] = None
    chunks: Optional[List['StrawberryDocumentInput']] = None
    matches: Optional[List['StrawberryDocumentInput']] = None
