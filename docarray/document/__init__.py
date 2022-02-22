from typing import overload, Dict, Optional, List, TYPE_CHECKING, Sequence

from .data import DocumentData
from .mixins import AllMixins
from ..base import BaseDCType

if TYPE_CHECKING:
    from ..types import ArrayType, StructValueType, DocumentContentType


class Document(AllMixins, BaseDCType):
    _data_class = DocumentData
    _unresolved_fields_dest = 'tags'

    @overload
    def __init__(self):
        """Create an empty Document."""
        ...

    @overload
    def __init__(self, _obj: Optional['Document'] = None, copy: bool = False):
        ...

    @overload
    def __init__(
        self,
        _obj: Optional[Dict],
        copy: bool = False,
        field_resolver: Optional[Dict[str, str]] = None,
        unknown_fields_handler: str = 'catch',
    ):
        ...

    @overload
    def __init__(self, blob: Optional[bytes] = None, **kwargs):
        """Create a Document with binary content."""
        ...

    @overload
    def __init__(self, tensor: Optional['ArrayType'] = None, **kwargs):
        """Create a Document with NdArray-like content."""
        ...

    @overload
    def __init__(self, text: Optional[str] = None, **kwargs):
        """Create a Document with string content."""
        ...

    @overload
    def __init__(self, uri: Optional[str] = None, **kwargs):
        """Create a Document with content from a URI."""
        ...

    @overload
    def __init__(
        self,
        parent_id: Optional[str] = None,
        granularity: Optional[int] = None,
        adjacency: Optional[int] = None,
        blob: Optional[bytes] = None,
        tensor: Optional['ArrayType'] = None,
        mime_type: Optional[str] = None,
        text: Optional[str] = None,
        content: Optional['DocumentContentType'] = None,
        weight: Optional[float] = None,
        uri: Optional[str] = None,
        tags: Optional[Dict[str, 'StructValueType']] = None,
        offset: Optional[float] = None,
        location: Optional[List[float]] = None,
        embedding: Optional['ArrayType'] = None,
        modality: Optional[str] = None,
        evaluations: Optional[Dict[str, Dict[str, 'StructValueType']]] = None,
        scores: Optional[Dict[str, Dict[str, 'StructValueType']]] = None,
        chunks: Optional[Sequence['Document']] = None,
        matches: Optional[Sequence['Document']] = None,
    ):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
