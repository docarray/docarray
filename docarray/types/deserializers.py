from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_deserializer(doc: 'Document'):
    if 'image_type' in doc._metadata:
        if doc._metadata['image_type'] == 'uri':
            return doc._metadata['image_uri']
        elif doc._metadata['image_type'] == 'PIL':
            from PIL.Image import Image

            return Image.fromarray(doc.tensor)
        elif doc._metadata['image_type'] == 'ndarray':
            return doc.tensor
    else:
        raise ValueError('Invalid image Document')


def text_deserializer(doc: 'Document'):
    return doc.text


def audio_deserializer(doc: 'Document'):
    from PIL.Image import Image

    return Image.fromarray(doc.tensor)
