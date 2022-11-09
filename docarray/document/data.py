import mimetypes
import os
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

from docarray.math.ndarray import check_arraylike_equality

if TYPE_CHECKING:  # pragma: no cover
    from docarray.score import NamedScore
    from docarray import DocumentArray, Document
    from docarray.typing import ArrayType, StructValueType, DocumentContentType

default_values = dict(
    granularity=0,
    adjacency=0,
    parent_id='',
    blob=b'',
    text='',
    weight=0.0,
    uri='',
    mime_type='',
    tags=dict,
    _metadata=dict,
    offset=0.0,
    location=list,
    modality='',
    evaluations='Dict[str, NamedScore]',
    scores='Dict[str, NamedScore]',
    chunks='ChunkArray',
    matches='MatchArray',
    timestamps=dict,
)

_all_mime_types = set(mimetypes.types_map.values())


def _is_not_empty(attribute, value):
    if value is not None:
        if attribute not in default_values:
            return True
        else:
            dv = default_values[attribute]
            if dv in (
                'ChunkArray',
                'MatchArray',
                'DocumentArray',
                list,
                dict,
                'Dict[str, NamedScore]',
            ):
                if value:
                    return True
            elif value != dv:
                return True
    return False


@dataclass(unsafe_hash=True, eq=False)
class DocumentData:
    _reference_doc: 'Document' = field(hash=False, compare=False)
    id: str = field(default_factory=lambda: os.urandom(16).hex())
    parent_id: Optional[str] = None
    granularity: Optional[int] = None
    adjacency: Optional[int] = None
    blob: Optional[bytes] = None
    tensor: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    mime_type: Optional[str] = None  # must be put in front of `text` `content`
    text: Optional[str] = None
    content: Optional['DocumentContentType'] = None
    weight: Optional[float] = None
    uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    _metadata: Optional[Dict[str, 'StructValueType']] = None
    offset: Optional[float] = None
    location: Optional[List[float]] = None
    embedding: Optional['ArrayType'] = field(default=None, hash=False, compare=False)
    modality: Optional[str] = None
    evaluations: Optional[Dict[str, Union['NamedScore', Dict]]] = None
    scores: Optional[Dict[str, Union['NamedScore', Dict]]] = None
    chunks: Optional['DocumentArray'] = None
    matches: Optional['DocumentArray'] = None

    @property
    def _non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self):
            f_name = f.name
            if not f_name.startswith('_') or f_name == '_metadata':
                v = getattr(self, f_name)
                if _is_not_empty(f_name, v):
                    r.append(f_name)

        return tuple(r)

    def _set_default_value_if_none(self, key):
        if getattr(self, key) is None:
            v = default_values.get(key, None)
            if v is not None:
                if v == 'DocumentArray':
                    from docarray import DocumentArray

                    setattr(self, key, DocumentArray())
                elif v == 'ChunkArray':
                    from docarray.array.chunk import ChunkArray

                    setattr(
                        self, key, ChunkArray(None, reference_doc=self._reference_doc)
                    )
                elif v == 'MatchArray':
                    from docarray.array.match import MatchArray

                    setattr(
                        self, key, MatchArray(None, reference_doc=self._reference_doc)
                    )
                elif v == 'Dict[str, NamedScore]':
                    from docarray.score import NamedScore

                    setattr(self, key, defaultdict(NamedScore))
                else:
                    setattr(self, key, v() if callable(v) else v)

    @staticmethod
    def _embedding_eq(array1: 'ArrayType', array2: 'ArrayType'):

        if array1 is None and array2 is None:
            return True

        if type(array1) == type(array2):
            return check_arraylike_equality(array1, array2)
        else:
            return False

    @staticmethod
    def _tensor_eq(array1: 'ArrayType', array2: 'ArrayType'):
        DocumentData._embedding_eq(array1, array2)

    def __eq__(self, other):

        self_non_empty_fields = self._non_empty_fields
        other_non_empty_fields = other._non_empty_fields

        if other_non_empty_fields != self_non_empty_fields:
            return False

        for key in self_non_empty_fields:

            if hasattr(self, f'_{key}_eq'):

                if hasattr(DocumentData, f'_{key}_eq'):
                    are_equal = getattr(DocumentData, f'_{key}_eq')(
                        getattr(self, key), getattr(other, key)
                    )
                    print(
                        f'are_equal( {getattr(self, key)}, {getattr(other, key)}) ---> {are_equal}'
                    )
                    if are_equal == False:
                        return False
            else:
                if getattr(self, key) != getattr(other, key):
                    return False
        return True
