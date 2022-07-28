from typing import overload, Dict, Optional, List, TYPE_CHECKING, Sequence, Any

from docarray.document.data import DocumentData
from docarray.document.mixins import AllMixins
from docarray.base import BaseDCType
from docarray.math.ndarray import detach_tensor_if_present

if TYPE_CHECKING:
    from docarray.typing import ArrayType, StructValueType, DocumentContentType


class Document(AllMixins, BaseDCType):
    """Document is the basic data type in DocArray.
    A Document is a container for any kind of data, be it text, image, audio, video, or 3D meshes.

    You can initialize a Document object with given attributes:

    .. code-block:: python

        from docarray import Document
        import numpy

        d1 = Document(text='hello')
        d3 = Document(tensor=numpy.array([1, 2, 3]))
        d4 = Document(
            uri='https://jina.ai',
            mime_type='text/plain',
            granularity=1,
            adjacency=3,
            tags={'foo': 'bar'},
        )

    Documents support a :ref:`nested structure <recursive-nested-document>`, which can also be specified during construction:

    .. code-block:: python

        d = Document(
            id='d0',
            chunks=[Document(id='d1', chunks=Document(id='d2'))],
            matches=[Document(id='d3')],
        )

    A Document can embed its contents using the :meth:`embed` method and a provided embedding model:

    .. code-block:: python

        import torchvision

        q = (
            Document(uri='/Users/usr/path/to/image.jpg')
            .load_uri_to_image_tensor()
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )
        model = torchvision.models.resnet50(pretrained=True)
        q.embed(model)

    Multiple Documents can be organized into a :class:`~docarray.array.document.DocumentArray`.

    .. seealso::
        For further details, see our :ref:`user guide <document>`.
    """

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
        """Create a Document from a `docarray.dataclass` instance"""
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
