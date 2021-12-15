from typing import TYPE_CHECKING, Dict, List, Optional
import copy as cp
from .base import BaseDocument, DocumentData
from .mixins import AllMixins

if TYPE_CHECKING:
    from .score import NamedScore
    from ..typing import ArrayType, DocumentContentType, StructValueType
    from datetime import datetime
    from .. import DocumentArray


class Document(AllMixins, BaseDocument):
    _all_doc_content_keys = {'content', 'blob', 'text', 'buffer'}

    def __init__(self, obj: Optional['Document'] = None,
                 copy: bool = False, **kwargs):
        self._pb_body = None
        if isinstance(obj, Document):
            if copy:
                self._pb_body = cp.deepcopy(obj._pb_body)
            else:
                self._pb_body = obj._pb_body
        if kwargs:
            # check if there are mutually exclusive content fields
            if len(self._all_doc_content_keys.intersection(kwargs.keys())) > 1:
                raise ValueError(
                    f'Document content fields are mutually exclusive, please provide only one of {self._all_doc_content_keys}'
                )
            self._pb_body = DocumentData(**kwargs)
        if obj is None and not kwargs:
            self._pb_body = DocumentData()

    def _clear_content(self):
        self._pb_body.text = None
        self._pb_body.blob = None
        self._pb_body.buffer = None

    @property
    def id(self) -> str:
        return self._pb_body.id

    @id.setter
    def id(self, value: str):
        self._pb_body.id = value

    @property
    def granularity(self) -> Optional[int]:
        return self._pb_body.granularity

    @granularity.setter
    def granularity(self, value: Optional[int]):
        self._pb_body.granularity = value

    @property
    def adjacency(self) -> Optional[int]:
        return self._pb_body.adjacency

    @adjacency.setter
    def adjacency(self, value: Optional[int]):
        self._pb_body.adjacency = value

    @property
    def parent_id(self) -> Optional[str]:
        return self._pb_body.parent_id

    @parent_id.setter
    def parent_id(self, value: Optional[str]):
        self._pb_body.parent_id = value

    @property
    def buffer(self) -> Optional[bytes]:
        return self._pb_body.buffer

    @buffer.setter
    def buffer(self, value: Optional[bytes]):
        self._clear_content()
        self._pb_body.buffer = value

    @property
    def blob(self) -> Optional['ArrayType']:
        return self._pb_body.blob

    @blob.setter
    def blob(self, value: Optional['ArrayType']):
        self._clear_content()
        self._pb_body.blob = value

    @property
    def text(self) -> Optional[str]:
        return self._pb_body.text

    @text.setter
    def text(self, value: Optional[str]):
        self._clear_content()
        self._pb_body.text = value

    @property
    def content(self) -> Optional['DocumentContentType']:
        return self._pb_body.content

    @content.setter
    def content(self, value: Optional['DocumentContentType']):
        if isinstance(value, bytes):
            self.buffer = value
        elif isinstance(value, str):
            self.text = value
        else:
            self.blob = value

    @property
    def weight(self) -> Optional[float]:
        return self._pb_body.weight

    @weight.setter
    def weight(self, value: Optional[float]):
        self._pb_body.weight = value

    @property
    def uri(self) -> Optional[str]:
        return self._pb_body.uri

    @uri.setter
    def uri(self, value: Optional[str]):
        self._pb_body.uri = value

    @property
    def mime_type(self) -> Optional[str]:
        return self._pb_body.mime_type

    @mime_type.setter
    def mime_type(self, value: Optional[str]):
        self._pb_body.mime_type = value

    @property
    def tags(self) -> Optional[Dict[str, 'StructValueType']]:
        return self._pb_body.tags

    @tags.setter
    def tags(self, value: Optional[Dict[str, 'StructValueType']]):
        self._pb_body.tags = value

    @property
    def offset(self) -> Optional[float]:
        return self._pb_body.offset

    @offset.setter
    def offset(self, value: Optional[float]):
        self._pb_body.offset = value

    @property
    def location(self) -> Optional[List[float]]:
        return self._pb_body.location

    @location.setter
    def location(self, value: Optional[List[float]]):
        self._pb_body.location = value

    @property
    def embedding(self) -> Optional['ArrayType']:
        return self._pb_body.embedding

    @embedding.setter
    def embedding(self, value: Optional['ArrayType']):
        self._pb_body.embedding = value

    @property
    def modality(self) -> Optional[str]:
        return self._pb_body.modality

    @modality.setter
    def modality(self, value: Optional[str]):
        self._pb_body.modality = value

    @property
    def evaluations(self) -> Optional[Dict[str, 'NamedScore']]:
        return self._pb_body.evaluations

    @evaluations.setter
    def evaluations(self, value: Optional[Dict[str, 'NamedScore']]):
        self._pb_body.evaluations = value

    @property
    def scores(self) -> Optional[Dict[str, 'NamedScore']]:
        return self._pb_body.scores

    @scores.setter
    def scores(self, value: Optional[Dict[str, 'NamedScore']]):
        self._pb_body.scores = value

    @property
    def chunks(self) -> Optional['DocumentArray']:
        if self._pb_body.chunks is None:
            self._pb_body.chunks = DocumentArray()
        return self._pb_body.chunks

    @chunks.setter
    def chunks(self, value: Optional['DocumentArray']):
        self._pb_body.chunks = value

    @property
    def matches(self) -> Optional['DocumentArray']:
        return self._pb_body.matches

    @matches.setter
    def matches(self, value: Optional['DocumentArray']):
        self._pb_body.matches = value

    @property
    def timestamps(self) -> Optional[Dict[str, 'datetime']]:
        return self._pb_body.timestamps

    @timestamps.setter
    def timestamps(self, value: Optional[Dict[str, 'datetime']]):
        self._pb_body.timestamps = value
