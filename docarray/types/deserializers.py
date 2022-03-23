import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_deserializer(attribute_name, doc: 'Document'):
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


def text_deserializer(attribute_name, doc: 'Document'):
    return doc.text


def audio_deserializer(attribute_name, doc: 'Document'):
    from PIL.Image import Image

    return Image.fromarray(doc.tensor)


def json_deserializer(attribute_name, doc: 'Document'):
    if doc._metadata['json_type'] == 'str':
        return json.dumps(doc.tags[attribute_name])
    else:
        return doc.tags[attribute_name]
