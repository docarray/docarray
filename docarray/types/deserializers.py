from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_deserializer(doc: 'Document'):
    return doc.tensor


def text_deserializer(doc: 'Document'):
    return doc.text


def pil_image_deserializer(img, doc: 'Document'):
    from PIL.Image import Image

    return Image.fromarray(doc.tensor)
