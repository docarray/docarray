from .attribute import GetAttributesMixin
from .audio import AudioDataMixin
from .blob import BlobDataMixin
from .content import ContentPropertyMixin
from .convert import ConvertMixin
from .dump import UriFileMixin
from .featurehash import FeatureHashMixin
from .image import ImageDataMixin
from .mesh import MeshDataMixin
from .plot import PlotMixin
from .porting import PortingMixin
from .property import PropertyMixin
from .protobuf import ProtobufMixin
from .pydantic import PydanticMixin
from .sugar import SingletonSugarMixin
from .text import TextDataMixin
from .video import VideoDataMixin


class AllMixins(
    ProtobufMixin,
    PydanticMixin,
    PropertyMixin,
    ContentPropertyMixin,
    ConvertMixin,
    AudioDataMixin,
    ImageDataMixin,
    TextDataMixin,
    MeshDataMixin,
    VideoDataMixin,
    BlobDataMixin,
    PlotMixin,
    UriFileMixin,
    SingletonSugarMixin,
    PortingMixin,
    FeatureHashMixin,
    GetAttributesMixin,
):
    """All plugins that can be used in :class:`Document`. """

    ...
