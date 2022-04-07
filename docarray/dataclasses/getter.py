from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_getter(doc: 'Document'):
    if doc._metadata['image_type'] == 'uri':
        return doc.uri
    elif doc._metadata['image_type'] == 'PIL':
        from PIL import Image

        return Image.fromarray(doc.tensor)
    elif doc._metadata['image_type'] == 'ndarray':
        return doc.tensor


def text_getter(doc: 'Document'):
    return doc.text


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
