from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar

import numpy as np
from pydantic.validators import bytes_validator

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.image_url import _move_channel_axis, _to_image_tensor

if TYPE_CHECKING:
    from pydantic.fields import BaseConfig, ModelField
T = TypeVar('T', bound='ImageBytes')


@_register_proto(proto_type_name='image_bytes')
class ImageBytes(bytes):
    """
    Bytes that store an image
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:

        value = bytes_validator(value)
        return cls(value)

    def load(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis_layout: Tuple[str, str, str] = ('H', 'W', 'C'),
    ) -> np.ndarray:
        """
        Load the data from the bytes into a numpy.ndarray image tensor

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            from docarray.typing import ImageUrl
            import numpy as np


            class MyDoc(BaseDocument):
                img_url: ImageUrl


            doc = MyDoc(
                img_url="https://upload.wikimedia.org/wikipedia/commons/8/80/"
                "Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg"
            )

            img_tensor = doc.img_url.load()
            assert isinstance(img_tensor, np.ndarray)

            img_tensor = doc.img_url.load(height=224, width=224)
            assert img_tensor.shape == (224, 224, 3)

            layout = ('C', 'W', 'H')
            img_tensor = doc.img_url.load(height=100, width=200, axis_layout=layout)
            assert img_tensor.shape == (3, 200, 100)


        :param width: width of the image tensor.
        :param height: height of the image tensor.
        :param axis_layout: ordering of the different image axes.
            'H' = height, 'W' = width, 'C' = color channel
        :return: np.ndarray representing the image as RGB values
        """

        tensor = _to_image_tensor(BytesIO(self), width=width, height=height)
        return _move_channel_axis(tensor, axis_layout=axis_layout)
