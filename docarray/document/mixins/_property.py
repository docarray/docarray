# auto-generated from /Users/hanxiao/Documents/docarray/scripts/gen_doc_property_mixin.py
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ...score import NamedScore
    from ...array.match import MatchArray
    from ...array.chunk import ChunkArray
    from ... import DocumentArray
    from ...types import ArrayType, StructValueType, DocumentContentType


class _PropertyMixin:
    @property
    def id(self) -> str:
        self._data._set_default_value_if_none('id')
        return self._data.id

    @id.setter
    def id(self, value: str):
        self._data.id = value

    @property
    def parent_id(self) -> Optional[str]:
        self._data._set_default_value_if_none('parent_id')
        return self._data.parent_id

    @parent_id.setter
    def parent_id(self, value: str):
        self._data.parent_id = value

    @property
    def granularity(self) -> Optional[int]:
        self._data._set_default_value_if_none('granularity')
        return self._data.granularity

    @granularity.setter
    def granularity(self, value: int):
        self._data.granularity = value

    @property
    def adjacency(self) -> Optional[int]:
        self._data._set_default_value_if_none('adjacency')
        return self._data.adjacency

    @adjacency.setter
    def adjacency(self, value: int):
        self._data.adjacency = value

    @property
    def blob(self) -> Optional[bytes]:
        self._data._set_default_value_if_none('blob')
        return self._data.blob

    @blob.setter
    def blob(self, value: bytes):
        self._data.blob = value

    @property
    def tensor(self) -> Optional['ArrayType']:
        self._data._set_default_value_if_none('tensor')
        return self._data.tensor

    @tensor.setter
    def tensor(self, value: 'ArrayType'):
        self._data.tensor = value

    @property
    def mime_type(self) -> Optional[str]:
        self._data._set_default_value_if_none('mime_type')
        return self._data.mime_type

    @mime_type.setter
    def mime_type(self, value: str):
        self._data.mime_type = value

    @property
    def text(self) -> Optional[str]:
        self._data._set_default_value_if_none('text')
        return self._data.text

    @text.setter
    def text(self, value: str):
        self._data.text = value

    @property
    def content(self) -> Optional['DocumentContentType']:
        self._data._set_default_value_if_none('content')
        return self._data.content

    @content.setter
    def content(self, value: 'DocumentContentType'):
        self._data.content = value

    @property
    def weight(self) -> Optional[float]:
        self._data._set_default_value_if_none('weight')
        return self._data.weight

    @weight.setter
    def weight(self, value: float):
        self._data.weight = value

    @property
    def uri(self) -> Optional[str]:
        self._data._set_default_value_if_none('uri')
        return self._data.uri

    @uri.setter
    def uri(self, value: str):
        self._data.uri = value

    @property
    def tags(self) -> Optional[Dict[str, 'StructValueType']]:
        self._data._set_default_value_if_none('tags')
        return self._data.tags

    @tags.setter
    def tags(self, value: Dict[str, 'StructValueType']):
        self._data.tags = value

    @property
    def offset(self) -> Optional[float]:
        self._data._set_default_value_if_none('offset')
        return self._data.offset

    @offset.setter
    def offset(self, value: float):
        self._data.offset = value

    @property
    def location(self) -> Optional[List[float]]:
        self._data._set_default_value_if_none('location')
        return self._data.location

    @location.setter
    def location(self, value: List[float]):
        self._data.location = value

    @property
    def embedding(self) -> Optional['ArrayType']:
        self._data._set_default_value_if_none('embedding')
        return self._data.embedding

    @embedding.setter
    def embedding(self, value: 'ArrayType'):
        self._data.embedding = value

    @property
    def modality(self) -> Optional[str]:
        self._data._set_default_value_if_none('modality')
        return self._data.modality

    @modality.setter
    def modality(self, value: str):
        self._data.modality = value

    @property
    def evaluations(self) -> Optional[Dict[str, 'NamedScore']]:
        self._data._set_default_value_if_none('evaluations')
        return self._data.evaluations

    @evaluations.setter
    def evaluations(self, value: Dict[str, 'NamedScore']):
        self._data.evaluations = value

    @property
    def scores(self) -> Optional[Dict[str, 'NamedScore']]:
        self._data._set_default_value_if_none('scores')
        return self._data.scores

    @scores.setter
    def scores(self, value: Dict[str, 'NamedScore']):
        self._data.scores = value

    @property
    def chunks(self) -> Optional['ChunkArray']:
        self._data._set_default_value_if_none('chunks')
        return self._data.chunks

    @chunks.setter
    def chunks(self, value: 'DocumentArray'):
        self._data.chunks = value

    @property
    def matches(self) -> Optional['MatchArray']:
        self._data._set_default_value_if_none('matches')
        return self._data.matches

    @matches.setter
    def matches(self, value: 'DocumentArray'):
        self._data.matches = value
