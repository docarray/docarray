from enum import Enum


class DocumentMetadata(str, Enum):
    MULTI_MODAL_SCHEMA = 'multi_modal_schema'
    IMAGE_TYPE = 'image_type'
    IMAGE_URI = 'image_uri'
    AUDIO_TYPE = 'audio_type'
    VIDEO_TYPE = 'video_type'
    MESH_TYPE = 'mesh_type'
    JSON_TYPE = 'json_type'
    BLOB_TYPE = 'blob_type'


class AttributeType(str, Enum):
    DOCUMENT = 'document'
    PRIMITIVE = 'primitive'
    ITERABLE_PRIMITIVE = 'iterable_primitive'
    ITERABLE_DOCUMENT = 'iterable_document'
    NESTED = 'nested'
    ITERABLE_NESTED = 'iterable_nested'


class ImageType(str, Enum):
    PIL = 'PIL'
    URI = 'uri'
    NDARRAY = 'ndarray'
