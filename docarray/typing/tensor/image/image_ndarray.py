from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.ndarray import NdArray

MAX_INT_16 = 2**15


@_register_proto(proto_type_name='image_ndarray')
class ImageNdArray(AbstractImageTensor, NdArray):
    """
    Subclass of NdArray, to represent an image tensor.
    Adds image-specific features to the tensor.
    For instance the ability convert the tensor back to image bytes which are
    optimized to send over the wire


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import ImageNdArray, ImageUrl
        import numpy as np


        class MyImageDoc(Document):
            title: str
            tensor: Optional[ImageNdArray]
            url: Optional[ImageUrl]
            bytes: Optional[bytes]


        # from url
        doc = MyImageDoc(
            title='my_second_audio_doc',
            url='https://an.image.png',
        )

        doc.tensor = doc.url.load()

        doc.bytes = doc.tensor.to_bytes()

    """

    ...
