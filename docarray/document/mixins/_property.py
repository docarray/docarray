# auto-generated from /Users/hanxiao/Documents/docarray/scripts/gen_doc_property_mixin.py
from typing import TYPE_CHECKING, Dict, List, Union

if TYPE_CHECKING:
    from ...score import NamedScore
    from ...array.match import MatchArray
    from ...array.chunk import ChunkArray
    from ... import DocumentArray
    from ...types import ArrayType, StructValueType, DocumentContentType


class _PropertyMixin:
    @property
    def id(self) -> str:
        return self._data.id

    @id.setter
    def id(self, value: str):
        self._data.id = value

    @property
    def parent_id(self) -> str:
        return self._data.parent_id

    @parent_id.setter
    def parent_id(self, value: str):
        self._data.parent_id = value

    @property
    def granularity(self) -> int:
        return self._data.granularity

    @granularity.setter
    def granularity(self, value: int):
        self._data.granularity = value

    @property
    def adjacency(self) -> int:
        return self._data.adjacency

    @adjacency.setter
    def adjacency(self, value: int):
        self._data.adjacency = value

    @property
    def blob(self) -> bytes:
        return self._data.blob

    @blob.setter
    def blob(self, value: bytes):
        self._data.blob = value

    @property
    def tensor(self) -> 'ArrayType':
        return self._data.tensor

    @tensor.setter
    def tensor(self, value: 'ArrayType'):
        self._data.tensor = value

    @property
    def mime_type(self) -> str:
        return self._data.mime_type

    @mime_type.setter
    def mime_type(self, value: str):
        self._data.mime_type = value

    @property
    def text(self) -> str:
        return self._data.text

    @text.setter
    def text(self, value: str):
        self._data.text = value

    @property
    def content(self) -> 'DocumentContentType':
        return self._data.content

    @content.setter
    def content(self, value: 'DocumentContentType'):
        self._data.content = value

    @property
    def weight(self) -> float:
        return self._data.weight

    @weight.setter
    def weight(self, value: float):
        self._data.weight = value

    @property
    def uri(self) -> str:
        return self._data.uri

    @uri.setter
    def uri(self, value: str):
        self._data.uri = value

    @property
    def tags(self) -> Dict[str, 'StructValueType']:
        return self._data.tags

    @tags.setter
    def tags(self, value: Dict[str, 'StructValueType']):
        self._data.tags = value

    @property
    def offset(self) -> float:
        return self._data.offset

    @offset.setter
    def offset(self, value: float):
        self._data.offset = value

    @property
    def location(self) -> List[float]:
        return self._data.location

    @location.setter
    def location(self, value: List[float]):
        self._data.location = value

    @property
    def embedding(self) -> 'ArrayType':
        return self._data.embedding

    @embedding.setter
    def embedding(self, value: 'ArrayType'):
        self._data.embedding = value

    @property
    def modality(self) -> str:
        return self._data.modality

    @modality.setter
    def modality(self, value: str):
        self._data.modality = value

    @property
    def evaluations(self) -> Dict[str, Union['NamedScore', Dict]]:
        return self._data.evaluations

    @evaluations.setter
    def evaluations(self, value: Dict[str, Union['NamedScore', Dict]]):
        self._data.evaluations = value

    @property
    def scores(self) -> Dict[str, Union['NamedScore', Dict]]:
        return self._data.scores

    @scores.setter
    def scores(self, value: Dict[str, Union['NamedScore', Dict]]):
        self._data.scores = value

    @property
    def chunks(self) -> 'ChunkArray':
        return self._data.chunks

    @chunks.setter
    def chunks(self, value: 'DocumentArray'):
        self._data.chunks = value

    @property
    def matches(self) -> 'MatchArray':
        return self._data.matches

    @matches.setter
    def matches(self, value: 'DocumentArray'):
        self._data.matches = value
