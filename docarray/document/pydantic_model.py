from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

from pydantic import BaseModel, validator

from docarray.math.ndarray import to_list

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import ArrayType

# this order must be preserved: https://pydantic-docs.helpmanual.io/usage/types/#unions
_ProtoValueType = Optional[Union[bool, float, str, list, dict]]
_StructValueType = Union[
    _ProtoValueType, List[_ProtoValueType], Dict[str, _ProtoValueType]
]
_MetadataType = Dict[str, _StructValueType]


def _convert_ndarray_to_list(v: 'ArrayType'):
    if v is not None:
        return to_list(v)


class _NamedScore(BaseModel):
    value: Optional[float] = None
    op_name: Optional[str] = None
    description: Optional[str] = None
    ref_id: Optional[str] = None


class _MetadataModel(BaseModel):
    metadata: _MetadataType


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

    def __init__(self, **data):
        super().__init__(**data)
        # underscore attributes need to be set and validated manually
        _metadata = data.get('_metadata', None)
        if _metadata is not None:
            _md_model = _MetadataModel(metadata=_metadata)  # validate _metadata
            object.__setattr__(self, '_metadata', _md_model.metadata)


PydanticDocument.update_forward_refs()

PydanticDocumentArray = List[PydanticDocument]
