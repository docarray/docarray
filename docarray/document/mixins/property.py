# auto-generated from /Users/hanxiao/Documents/docarray/scripts/generate_property.py
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..score import NamedScore
    from ... import DocumentArray
    from ...typing import ArrayType, StructValueType
    from datetime import datetime


class PropertyMixin:

    @property
    def non_empty_fields(self) -> Tuple[str]:
        """Get all non-emtpy fields of this :class:`Document`.

        Non-empty fields are the fields with not-`None` and not-default values.

        :return: field names in a tuple.
        """
        return self._doc_data.non_empty_fields
    
    @property
    def id(self) -> str:
        self._doc_data._set_default_value_if_none('id')
        return self._doc_data.id

    @id.setter
    def id(self, value: str):
        self._doc_data.id = value
        
    @property
    def granularity(self) -> Optional[int]:
        self._doc_data._set_default_value_if_none('granularity')
        return self._doc_data.granularity

    @granularity.setter
    def granularity(self, value: Optional[int]):
        self._doc_data.granularity = value
        
    @property
    def adjacency(self) -> Optional[int]:
        self._doc_data._set_default_value_if_none('adjacency')
        return self._doc_data.adjacency

    @adjacency.setter
    def adjacency(self, value: Optional[int]):
        self._doc_data.adjacency = value
        
    @property
    def parent_id(self) -> Optional[str]:
        self._doc_data._set_default_value_if_none('parent_id')
        return self._doc_data.parent_id

    @parent_id.setter
    def parent_id(self, value: Optional[str]):
        self._doc_data.parent_id = value
        
    @property
    def buffer(self) -> Optional[bytes]:
        self._doc_data._set_default_value_if_none('buffer')
        return self._doc_data.buffer

    @buffer.setter
    def buffer(self, value: Optional[bytes]):
        self._doc_data.buffer = value
        
    @property
    def blob(self) -> Optional['ArrayType']:
        self._doc_data._set_default_value_if_none('blob')
        return self._doc_data.blob

    @blob.setter
    def blob(self, value: Optional['ArrayType']):
        self._doc_data.blob = value
        
    @property
    def text(self) -> Optional[str]:
        self._doc_data._set_default_value_if_none('text')
        return self._doc_data.text

    @text.setter
    def text(self, value: Optional[str]):
        self._doc_data.text = value
        
    @property
    def weight(self) -> Optional[float]:
        self._doc_data._set_default_value_if_none('weight')
        return self._doc_data.weight

    @weight.setter
    def weight(self, value: Optional[float]):
        self._doc_data.weight = value
        
    @property
    def uri(self) -> Optional[str]:
        self._doc_data._set_default_value_if_none('uri')
        return self._doc_data.uri

    @uri.setter
    def uri(self, value: Optional[str]):
        self._doc_data.uri = value
        
    @property
    def mime_type(self) -> Optional[str]:
        self._doc_data._set_default_value_if_none('mime_type')
        return self._doc_data.mime_type

    @mime_type.setter
    def mime_type(self, value: Optional[str]):
        self._doc_data.mime_type = value
        
    @property
    def tags(self) -> Optional[Dict[str, 'StructValueType']]:
        self._doc_data._set_default_value_if_none('tags')
        return self._doc_data.tags

    @tags.setter
    def tags(self, value: Optional[Dict[str, 'StructValueType']]):
        self._doc_data.tags = value
        
    @property
    def offset(self) -> Optional[float]:
        self._doc_data._set_default_value_if_none('offset')
        return self._doc_data.offset

    @offset.setter
    def offset(self, value: Optional[float]):
        self._doc_data.offset = value
        
    @property
    def location(self) -> Optional[List[float]]:
        self._doc_data._set_default_value_if_none('location')
        return self._doc_data.location

    @location.setter
    def location(self, value: Optional[List[float]]):
        self._doc_data.location = value
        
    @property
    def embedding(self) -> Optional['ArrayType']:
        self._doc_data._set_default_value_if_none('embedding')
        return self._doc_data.embedding

    @embedding.setter
    def embedding(self, value: Optional['ArrayType']):
        self._doc_data.embedding = value
        
    @property
    def modality(self) -> Optional[str]:
        self._doc_data._set_default_value_if_none('modality')
        return self._doc_data.modality

    @modality.setter
    def modality(self, value: Optional[str]):
        self._doc_data.modality = value
        
    @property
    def evaluations(self) -> Optional[Dict[str, 'NamedScore']]:
        self._doc_data._set_default_value_if_none('evaluations')
        return self._doc_data.evaluations

    @evaluations.setter
    def evaluations(self, value: Optional[Dict[str, 'NamedScore']]):
        self._doc_data.evaluations = value
        
    @property
    def scores(self) -> Optional[Dict[str, 'NamedScore']]:
        self._doc_data._set_default_value_if_none('scores')
        return self._doc_data.scores

    @scores.setter
    def scores(self, value: Optional[Dict[str, 'NamedScore']]):
        self._doc_data.scores = value
        
    @property
    def chunks(self) -> Optional['DocumentArray']:
        self._doc_data._set_default_value_if_none('chunks')
        return self._doc_data.chunks

    @chunks.setter
    def chunks(self, value: Optional['DocumentArray']):
        self._doc_data.chunks = value
        
    @property
    def matches(self) -> Optional['DocumentArray']:
        self._doc_data._set_default_value_if_none('matches')
        return self._doc_data.matches

    @matches.setter
    def matches(self, value: Optional['DocumentArray']):
        self._doc_data.matches = value
        
    @property
    def timestamps(self) -> Optional[Dict[str, 'datetime']]:
        self._doc_data._set_default_value_if_none('timestamps')
        return self._doc_data.timestamps

    @timestamps.setter
    def timestamps(self, value: Optional[Dict[str, 'datetime']]):
        self._doc_data.timestamps = value
        