from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

from pydantic import BaseModel, validator

from ..math.ndarray import to_list

if TYPE_CHECKING:
    from ..typing import ArrayType

# this order must be preserved: https://pydantic-docs.helpmanual.io/usage/types/#unions
_ProtoValueType = Optional[Union[bool, float, str, list, dict]]
_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]


def _convert_ndarray_to_list(v: 'ArrayType'):
    if v is not None:
        return to_list(v)


class _NamedScore(BaseModel):
    value: Optional[float] = None
    op_name: Optional[str] = None
    description: Optional[str] = None
    ref_id: Optional[str] = None


class PydanticDocument(BaseModel):
    id: Optional[str]
    parent_id: Optional[str]
    granularity: Optional[int]
    adjacency: Optional[int]
    blob: Optional[str]
    tensor: Optional[Any]
    mime_type: Optional[str]
    text: Optional[str]
    weight: Optional[float]
    uri: Optional[str]
    tags: Optional[Dict[str, '_StructValueType']]
    _metadata: Optional[Dict[str, '_StructValueType']]
    offset: Optional[float]
    location: Optional[List[float]]
    embedding: Optional[Any]
    modality: Optional[str]
    evaluations: Optional[Dict[str, '_NamedScore']]
    scores: Optional[Dict[str, '_NamedScore']]
    chunks: Optional[List['PydanticDocument']]
    matches: Optional[List['PydanticDocument']]

    _tensor2list = validator('tensor', allow_reuse=True)(_convert_ndarray_to_list)
    _embedding2list = validator('embedding', allow_reuse=True)(_convert_ndarray_to_list)

    class Config:
        smart_union = True


PydanticDocument.update_forward_refs()

PydanticDocumentArray = List[PydanticDocument]
