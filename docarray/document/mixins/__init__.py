from docarray.document.mixins.attribute import GetAttributesMixin
from docarray.document.mixins.audio import AudioDataMixin
from docarray.document.mixins.blob import BlobDataMixin
from docarray.document.mixins.content import ContentPropertyMixin
from docarray.document.mixins.convert import ConvertMixin
from docarray.document.mixins.dump import UriFileMixin
from docarray.document.mixins.featurehash import FeatureHashMixin
from docarray.document.mixins.image import ImageDataMixin
from docarray.document.mixins.mesh import MeshDataMixin
from docarray.document.mixins.multimodal import MultiModalMixin
from docarray.document.mixins.plot import PlotMixin
from docarray.document.mixins.porting import PortingMixin
from docarray.document.mixins.property import PropertyMixin
from docarray.document.mixins.protobuf import ProtobufMixin
from docarray.document.mixins.pydantic import PydanticMixin
from docarray.document.mixins.strawberry import StrawberryMixin
from docarray.document.mixins.sugar import SingletonSugarMixin
from docarray.document.mixins.text import TextDataMixin
from docarray.document.mixins.video import VideoDataMixin


class AllMixins(
    ProtobufMixin,
    PydanticMixin,
    StrawberryMixin,
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
    MultiModalMixin,
):
    """All plugins that can be used in :class:`Document`."""

    ...
