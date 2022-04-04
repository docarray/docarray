import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_getter(doc: 'Document', field_name: str):
    if 'image_type' in doc._metadata:
        if doc._metadata['image_type'] == 'uri':
            return doc._metadata['image_uri']
        elif doc._metadata['image_type'] == 'PIL':
            from PIL import Image

            return Image.fromarray(doc.tensor)
        elif doc._metadata['image_type'] == 'ndarray':
            return doc.tensor
    else:
        raise ValueError('Invalid image Document')


def text_getter(doc: 'Document', field_name: str):
    return doc.text


def audio_getter(doc: 'Document', field_name: str):
    from PIL import Image

    return Image.fromarray(doc.tensor)


def json_getter(doc: 'Document', field_name: str):
    if doc._metadata['json_type'] == 'str':
        return json.dumps(doc.tags[field_name])
    else:
        return doc.tags[field_name]
