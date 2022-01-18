import base64
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

from pydantic import BaseModel, validator

from ..math.ndarray import to_list

if TYPE_CHECKING:
    from ..types import ArrayType

# this order must be preserved: https://pydantic-docs.helpmanual.io/usage/types/#unions
_ProtoValueType = Optional[Union[bool, float, str]]
_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]


def _convert_ndarray_to_list(v: 'ArrayType'):
    if v is not None:
        return to_list(v)


class PydanticDocument(BaseModel):
    id: str
    parent_id: Optional[str]
    granularity: Optional[int]
    adjacency: Optional[int]
    blob: Optional[bytes]
    tensor: Optional[Any]
    mime_type: Optional[str]
    text: Optional[str]
    weight: Optional[float]
    uri: Optional[str]
    tags: Optional[Dict[str, '_StructValueType']]
    offset: Optional[float]
    location: Optional[List[float]]
    embedding: Optional[Any]
    modality: Optional[str]
    evaluations: Optional[Dict[str, Dict[str, '_StructValueType']]]
    scores: Optional[Dict[str, Dict[str, '_StructValueType']]]
    chunks: Optional[List['PydanticDocument']]
    matches: Optional[List['PydanticDocument']]

    _tensor2list = validator('tensor', allow_reuse=True)(_convert_ndarray_to_list)
    _embedding2list = validator('embedding', allow_reuse=True)(_convert_ndarray_to_list)

    @validator('blob')
    def _blob2base64(cls, v):
        if v is not None:
            if isinstance(v, bytes):
                return base64.b64encode(v).decode('utf8')
            else:
                raise ValueError('must be bytes')


PydanticDocument.update_forward_refs()

PydanticDocumentArray = List[PydanticDocument]
