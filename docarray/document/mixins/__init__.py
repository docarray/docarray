from .attribute import GetAttributesMixin
from .audio import AudioDataMixin
from .base import BaseDocumentMixin
from .buffer import BufferDataMixin
from .content import ContentPropertyMixin
from .convert import ConvertMixin
from .dump import DumpFileMixin
from .image import ImageDataMixin
from .mesh import MeshDataMixin
from .plot import PlotMixin
from .property import PropertyMixin
from .sugar import SingletonSugarMixin
from .text import TextDataMixin
from .video import VideoDataMixin


class AllMixins(
    BaseDocumentMixin,
    PropertyMixin,
    ContentPropertyMixin,
    ConvertMixin,
    AudioDataMixin,
    ImageDataMixin,
    TextDataMixin,
    MeshDataMixin,
    VideoDataMixin,
    BufferDataMixin,
    PlotMixin,
    DumpFileMixin,
    SingletonSugarMixin,
    GetAttributesMixin,
):
    """All plugins that can be used in :class:`Document`. """

    ...
