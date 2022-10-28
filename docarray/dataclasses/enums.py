from enum import Enum


class DocumentMetadata(str, Enum):
    IMAGE_TYPE = 'image_type'
    IMAGE_URI = 'image_uri'
    MULTI_MODAL_SCHEMA = 'multi_modal_schema'
    JSON_TYPE = 'json_type'
