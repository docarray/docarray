from typing import TYPE_CHECKING
from docarray.dataclasses.enums import DocumentMetadata, ImageType

if TYPE_CHECKING:  # pragma: no cover
    from docarray import Document


def image_getter(doc: 'Document'):
    if doc._metadata[DocumentMetadata.IMAGE_TYPE] == ImageType.URI:
        return doc.uri
    elif doc._metadata[DocumentMetadata.IMAGE_TYPE] == ImageType.PIL:
        from PIL import Image

        return Image.fromarray(doc.tensor)
    elif doc._metadata[DocumentMetadata.IMAGE_TYPE] == ImageType.NDARRAY:
        return doc.tensor


def text_getter(doc: 'Document'):
    return doc.text


def uri_getter(doc: 'Document'):
    return doc.uri


def audio_getter(doc: 'Document'):
    return doc.uri or doc.tensor


def video_getter(doc: 'Document'):
    return doc.uri or doc.tensor


def mesh_getter(doc: 'Document'):
    return doc.uri or doc.tensor


def tabular_getter(doc: 'Document'):
    return doc.uri


def blob_getter(doc: 'Document'):
    return doc.uri or doc.blob


def json_getter(doc: 'Document'):
    return doc.tags
