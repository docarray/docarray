from typing import TYPE_CHECKING

import numpy as np
from PIL.Image import Image

if TYPE_CHECKING:
    from docarray import Document


def image_serializer(inp, doc: 'Document'):
    if isinstance(inp, str):
        doc.uri = inp
        doc._metadata['image_type'] = 'uri'
        doc._metadata['image_uri'] = inp
        doc.load_uri_to_image_tensor()
    elif isinstance(inp, Image):
        doc.tensor = np.array(inp)
        doc._metadata['image_type'] = 'PIL'
    else:
        doc.tensor = inp
        doc._metadata['image_type'] = 'ndarray'
    doc.modality = 'image'


def text_serializer(text, doc: 'Document'):
    doc.text = text
    doc.modality = 'text'


def audio_serializer(uri, doc: 'Document'):
    import librosa

    audio, sr = librosa.load(uri)
    doc.tensor = audio
    doc._metadata['audio_sample_rate'] = sr
    doc._metadata['audio_uri'] = str(uri)
    doc.modality = 'audio'
