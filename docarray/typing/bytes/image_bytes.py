from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar

import numpy as np
from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.image_ndarray import ImageNdArray
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from pydantic.fields import BaseConfig, ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='ImageBytes')


@_register_proto(proto_type_name='image_bytes')
class ImageBytes(bytes, AbstractType):
    """
    Bytes that store an image and that can be load into an image tensor
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        value = bytes_validator(value)
        return cls(value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    def load_pil(
        self,
    ) -> 'PILImage.Image':
        """
        Load the image from the bytes into a `PIL.Image.Image` instance

        ---

        ```python
        from pydantic import parse_obj_as

        from docarray import BaseDoc
        from docarray.typing import ImageUrl

        img_url = "https://upload.wikimedia.org/wikipedia/commons/8/80/Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg"

        img_url = parse_obj_as(ImageUrl, img_url)
        img = img_url.load_pil()

        from PIL.Image import Image

        assert isinstance(img, Image)
        ```

        ---
        :return: a Pillow image
        """
        PIL = import_library('PIL', raise_error=True)  # noqa: F841
        from PIL import Image as PILImage

        return PILImage.open(BytesIO(self))

    def load(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis_layout: Tuple[str, str, str] = ('H', 'W', 'C'),
    ) -> ImageNdArray:
        """
        Load the image from the [`ImageBytes`][docarray.typing.ImageBytes] into an
        [`ImageNdArray`][docarray.typing.ImageNdArray].

        ---

        ```python
        from docarray import BaseDoc
        from docarray.typing import ImageNdArray, ImageUrl


        class MyDoc(BaseDoc):
            img_url: ImageUrl


        doc = MyDoc(
            img_url="https://upload.wikimedia.org/wikipedia/commons/8/80/"
            "Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg"
        )

        img_tensor = doc.img_url.load()
        assert isinstance(img_tensor, ImageNdArray)

        img_tensor = doc.img_url.load(height=224, width=224)
        assert img_tensor.shape == (224, 224, 3)

        layout = ('C', 'W', 'H')
        img_tensor = doc.img_url.load(height=100, width=200, axis_layout=layout)
        assert img_tensor.shape == (3, 200, 100)
        ```

        ---

        :param width: width of the image tensor.
        :param height: height of the image tensor.
        :param axis_layout: ordering of the different image axes.
            'H' = height, 'W' = width, 'C' = color channel
        :return: [`ImageNdArray`][docarray.typing.ImageNdArray] representing the image as RGB values
        """
        raw_img = self.load_pil()

        if width or height:
            new_width = width or raw_img.width
            new_height = height or raw_img.height
            raw_img = raw_img.resize((new_width, new_height))
        try:
            tensor = np.array(raw_img.convert('RGB'))
        except Exception:
            tensor = np.array(raw_img)

        img = self._move_channel_axis(tensor, axis_layout=axis_layout)
        return parse_obj_as(ImageNdArray, img)

    @staticmethod
    def _move_channel_axis(
        tensor: np.ndarray, axis_layout: Tuple[str, str, str] = ('H', 'W', 'C')
    ) -> np.ndarray:
        """Moves channel axis around."""
        channel_to_offset = {'H': 0, 'W': 1, 'C': 2}
        permutation = tuple(channel_to_offset[axis] for axis in axis_layout)
        return np.transpose(tensor, permutation)
