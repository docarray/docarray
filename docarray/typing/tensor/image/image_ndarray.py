from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.ndarray import NdArray

MAX_INT_16 = 2**15


@_register_proto(proto_type_name='image_ndarray')
class ImageNdArray(AbstractImageTensor, NdArray):
    """
    Subclass of NdArray, to represent an image tensor.
    Adds image-specific features to the tensor.


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import ImageNdArray, ImageUrl
        import numpy as np


        class MyAudioDoc(Document):
            title: str
            tensor: Optional[ImageNdArray]
            url: Optional[ImageUrl]


        # from url
        doc_2 = MyAudioDoc(
            title='my_second_audio_doc',
            url='https://an.image.png',
        )

        doc_2.audio_tensor = doc_2.url.load()

    """

    def bytes(self):
        tensor = (self * MAX_INT_16).astype('<h')
        return tensor.tobytes()
