from enum import Enum


class DocumentMetadata(str, Enum):
    IMAGE_TYPE = 'image_type'
    IMAGE_URI = 'image_uri'
    MULTI_MODAL_SCHEMA = 'multi_modal_schema'
    JSON_TYPE = 'json_type'


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
