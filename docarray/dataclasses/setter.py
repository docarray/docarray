from typing import TYPE_CHECKING
import numpy as np

from docarray.dataclasses.enums import DocumentMetadata, ImageType


if TYPE_CHECKING:  # pragma: no cover
    from docarray import Document


def image_setter(value) -> 'Document':
    from docarray import Document

    doc = Document(modality='image')

    if isinstance(value, str):
        doc.uri = value
        doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.URI
        doc.load_uri_to_image_tensor()
    elif isinstance(value, np.ndarray):
        doc.tensor = value
        doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.NDARRAY
    else:
        from PIL.Image import Image

        if isinstance(value, Image):
            doc.tensor = np.array(value)
            doc._metadata[DocumentMetadata.IMAGE_TYPE] = ImageType.PIL
    return doc


def text_setter(value) -> 'Document':
    from docarray import Document

    return Document(text=value, modality='text')


def uri_setter(value) -> 'Document':
    from docarray import Document

    return Document(uri=value)


def audio_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(
            tensor=value, _metadata={DocumentMetadata.AUDIO_TYPE: 'ndarray'}
        )
    else:
        return Document(
            uri=value, modality='audio', _metadata={DocumentMetadata.AUDIO_TYPE: 'uri'}
        ).load_uri_to_audio_tensor()


def video_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(
            tensor=value, _metadata={DocumentMetadata.VIDEO_TYPE: 'ndarray'}
        )
    else:
        return Document(
            uri=value, modality='video', _metadata={DocumentMetadata.VIDEO_TYPE: 'uri'}
        ).load_uri_to_video_tensor()


def mesh_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, np.ndarray):
        return Document(tensor=value, _metadata={DocumentMetadata.MESH_TYPE: 'ndarray'})
    else:
        return Document(
            uri=value, modality='mesh', _metadata={DocumentMetadata.MESH_TYPE: 'uri'}
        ).load_uri_to_point_cloud_tensor(1000)


def blob_setter(value) -> 'Document':
    from docarray import Document

    if isinstance(value, bytes):
        return Document(blob=value, _metadata={DocumentMetadata.BLOB_TYPE: 'bytes'})
    else:
        return Document(
            uri=value, _metadata={DocumentMetadata.BLOB_TYPE: 'uri'}
        ).load_uri_to_blob()


def json_setter(value) -> 'Document':
    from docarray import Document

    return Document(modality='json', tags=value)


def tabular_setter(value) -> 'Document':
    from docarray import Document, DocumentArray

    return Document(uri=value, chunks=DocumentArray.from_csv(value), modality='tabular')
