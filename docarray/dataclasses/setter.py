import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from docarray import Document


def image_setter(field_name: str, value) -> 'Document':
    from PIL.Image import Image
    from docarray import Document

    doc = Document(modality='image')

    if isinstance(value, str):
        doc.uri = value
        doc._metadata['image_type'] = 'uri'
        doc._metadata['image_uri'] = value
        doc.load_uri_to_image_tensor()
    elif isinstance(value, Image):
        doc.tensor = np.array(value)
        doc._metadata['image_type'] = 'PIL'
    else:
        doc.tensor = value
        doc._metadata['image_type'] = 'ndarray'
    return doc


def text_setter(field_name: str, value) -> 'Document':
    from docarray import Document

    return Document(text=value, modality='text')


def audio_setter(field_name: str, value) -> 'Document':
    import librosa
    from docarray import Document

    audio, sr = librosa.load(value)
    return Document(
        tensor=audio,
        _metadata={'audio_sample_rate': sr, 'audio_uri': str(value)},
        modality='audio',
    )


def json_setter(field_name: str, value) -> 'Document':
    from docarray import Document

    doc = Document()
    if isinstance(value, str):
        value = json.loads(value)
        doc._metadata['json_type'] = 'str'
    else:
        doc._metadata['json_type'] = 'dict'
    doc.tags[field_name] = value
    return doc
