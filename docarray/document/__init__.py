from typing import overload, Dict, Optional, List, TYPE_CHECKING, Sequence, Any

from .data import DocumentData
from .mixins import AllMixins
from ..base import BaseDCType
from ..math.ndarray import detach_tensor_if_present

if TYPE_CHECKING:
    from ..typing import ArrayType, StructValueType, DocumentContentType


class Document(AllMixins, BaseDCType):
    _data_class = DocumentData
    _unresolved_fields_dest = 'tags'
    _post_init_fields = (
        'text',
        'blob',
        'tensor',
        'content',
        'uri',
        'mime_type',
        'chunks',
        'matches',
    )

    @overload
    def __init__(self):
        """Create an empty Document."""
        ...

    @overload
    def __init__(self, _obj: Optional['Document'] = None, copy: bool = False):
        ...

    @overload
    def __init__(self, _obj: Optional[Any] = None):
        """Create a Document from a `docarray.dataclasses.dataclass` instance"""
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

    def __getstate__(self):
        state = self.__dict__.copy()

        for attribute in ['embedding', 'tensor']:
            if hasattr(self, attribute):
                setattr(
                    state['_data'],
                    attribute,
                    detach_tensor_if_present(getattr(state['_data'], attribute)),
                )

        return state
