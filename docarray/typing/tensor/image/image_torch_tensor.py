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


    ---

    ```python
    from typing import Optional

    from docarray import BaseDoc
    from docarray.typing import ImageTorchTensor, ImageUrl


    class MyImageDoc(BaseDoc):
        title: str
        tensor: Optional[ImageTorchTensor]
        url: Optional[ImageUrl]
        bytes: Optional[bytes]


    doc = MyImageDoc(
        title='my_second_image_doc',
        url="https://upload.wikimedia.org/wikipedia/commons/8/80/"
        "Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg",
    )

    doc.tensor = doc.url.load()
    doc.bytes = doc.tensor.to_bytes()
    ```

    ---
    """

    ...
