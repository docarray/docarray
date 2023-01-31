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


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import torch
        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import ImageTorchTensor, ImageUrl


        class MyImageDoc(Document):
            title: str
            image_tensor: Optional[ImageTorchTensor]
            url: Optional[ImageUrl]


        doc_1 = MyImageDoc(
            title='my_second_image_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_1.image_tensor = doc_2.url.load()

    """

    ...
