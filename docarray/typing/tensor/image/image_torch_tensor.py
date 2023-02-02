from typing import TypeVar

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode

T = TypeVar('T', bound='ImageTorchTensor')


@_register_proto(proto_type_name='image_torch_tensor')
class ImageTorchTensor(AbstractImageTensor, TorchTensor, metaclass=metaTorchAndNode):
    """
    Subclass of TorchTensor, to represent an image tensor.
    Adds image-specific features to the tensor.
    For instance the ability convert the tensor back to image bytes which are
    optimized to send over the wire


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import torch
        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import ImageTorchTensor, ImageUrl


        class MyImageDoc(Document):
            title: str
            tensor: Optional[ImageTorchTensor]
            url: Optional[ImageUrl]
            bytes: Optional[bytes]


        doc = MyImageDoc(
            title='my_second_image_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc.tensor = doc.url.load()
        doc.bytes = doc.tensor.to_bytes()
    """

    ...
