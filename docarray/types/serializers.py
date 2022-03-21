from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import Document


def image_serializer(img, doc: 'Document'):
    doc.tensor = img
    doc.modality = 'image'


def text_serializer(text, doc: 'Document'):
    doc.text = text
    doc.modality = 'text'


def pil_image_serializer(img, doc: 'Document'):
    import numpy as np

    doc.tensor = np.array(img)
    doc.modality = 'image'


def image_uri_serializer(uri, doc: 'Document'):
    doc.uri = uri
    doc.load_uri_to_image_tensor()
    doc.modality = 'image'


def audio_uri_serializer(uri, doc: 'Document'):
    import librosa

    audio, sr = librosa.load(uri)
    doc.tensor = audio
    doc._metadata['audio_sample_rate'] = sr
    doc.modality = 'audio'
