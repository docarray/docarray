import io
import struct
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.proto import NodeProto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper import _uri_to_blob

if TYPE_CHECKING:
    import PIL
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='ImageUrl')

IMAGE_FILE_FORMATS = ('png', 'jpeg', 'jpg')


class ImageUrl(AnyUrl):
    """ "
    URL to a .png, .jpeg, or .jpg file.
    Cane be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(image_url=str(self))

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:

        url = super().validate(value, field, config)  # basic url validation
        has_image_extension = any(url.endswith(ext) for ext in IMAGE_FILE_FORMATS)
        if not has_image_extension:
            raise ValueError(
                f'Image URL must have one of the following extensions:'
                f'{IMAGE_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def load(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis_layout: Tuple[str, str, str] = ('H', 'W', 'C'),
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray image tensor

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document
            from docarray.typing import ImageUrl
            import numpy as np


            class MyDoc(Document):
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
        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: np.ndarray representing the image as RGB values
        """

        buffer = _uri_to_blob(self, timeout=timeout)
        tensor = _to_image_tensor(io.BytesIO(buffer), width=width, height=height)
        return _move_channel_axis(tensor, axis_layout=axis_layout)

    def load_to_bytes(
        self,
        image_format: str = 'png',
        width: Optional[int] = None,
        height: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Load image at URL to bytes (buffer).

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document
            from docarray.typing import ImageUrl
            import numpy as np


            class MyDoc(Document):
                img_url: ImageUrl


            doc = MyDoc(
                img_url="https://upload.wikimedia.org/wikipedia/commons/8/80/"
                "Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg"
            )

            img_tensor = doc.img_url.load_to_bytes(image_format='jpg')
            assert isinstance(img_tensor, bytes)

        :param image_format: File format of the file located the the url.
            Supported formats are `png`, `jpg`, and `jpeg`.
        :param width: Before converting to bytes, resize the image to this width.
        :param height: Before converting to bytes, resize the image to this height.
        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: The image as bytes (buffer).
        """
        image_tensor = self.load(width=width, height=height, timeout=timeout)
        return _image_tensor_to_bytes(image_tensor, image_format=image_format)


def _image_tensor_to_bytes(arr: np.ndarray, image_format: str) -> bytes:
    """
    Convert image-ndarray to buffer bytes.

    :param arr: Data representations of the png.
    :param image_format: `png` or `jpeg`
    :return: Png in buffer bytes.
    """

    if image_format not in IMAGE_FILE_FORMATS:
        raise ValueError(
            f'image_format must be one of {IMAGE_FILE_FORMATS},'
            f'receiving `{image_format}`'
        )
    if image_format == 'jpg':
        image_format = 'jpeg'  # unify it to ISO standard

    arr = arr.astype(np.uint8).squeeze()

    if arr.ndim == 1:
        # note this should be only used for MNIST/FashionMNIST dataset,
        # because of the nature of these two datasets
        # no other image data should flattened into 1-dim array.
        image_bytes = _png_to_buffer_1d(arr, 28, 28)
    elif arr.ndim == 2:
        from PIL import Image

        im = Image.fromarray(arr).convert('L')
        image_bytes = _pillow_image_to_buffer(im, image_format=image_format.upper())
    elif arr.ndim == 3:
        from PIL import Image

        im = Image.fromarray(arr).convert('RGB')
        image_bytes = _pillow_image_to_buffer(im, image_format=image_format.upper())
    else:
        raise ValueError(
            f'{arr.shape} ndarray can not be converted into an image buffer.'
        )

    return image_bytes


def _png_to_buffer_1d(arr: np.ndarray, width: int, height: int) -> bytes:
    import zlib

    pixels = []
    for p in arr[::-1]:
        pixels.extend([p, p, p, 255])
    buf = bytearray(pixels)

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(
        b'\x00' + buf[span : span + width_byte_4]
        for span in range((height - 1) * width_byte_4, -1, -width_byte_4)
    )

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (
            struct.pack('!I', len(data))
            + chunk_head
            + struct.pack('!I', 0xFFFFFFFF & zlib.crc32(chunk_head))
        )

    png_bytes = b''.join(
        [
            b'\x89PNG\r\n\x1a\n',
            png_pack(b'IHDR', struct.pack('!2I5B', width, height, 8, 6, 0, 0, 0)),
            png_pack(b'IDAT', zlib.compress(raw_data, 9)),
            png_pack(b'IEND', b''),
        ]
    )

    return png_bytes


def _pillow_image_to_buffer(image: 'PIL.Image.Image', image_format: str) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image_format)
    img_bytes = img_byte_arr.getvalue()
    return img_bytes


def _move_channel_axis(
    tensor: np.ndarray, axis_layout: Tuple[str, str, str] = ('H', 'W', 'C')
) -> np.ndarray:
    """Moves channel axis around."""
    channel_to_offset = {'H': 0, 'W': 1, 'C': 2}
    permutation = tuple(channel_to_offset[axis] for axis in axis_layout)
    return np.transpose(tensor, permutation)


def _to_image_tensor(
    source: Union[str, bytes, io.BytesIO],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> 'np.ndarray':
    """
    Convert an image blob to tensor

    :param source: binary blob or file path
    :param width: the width of the image tensor.
    :param height: the height of the tensor.
    :return: image tensor
    """
    from PIL import Image as PILImage

    raw_img = PILImage.open(source)
    if width or height:
        new_width = width or raw_img.width
        new_height = height or raw_img.height
        raw_img = raw_img.resize((new_width, new_height))
    try:
        return np.array(raw_img.convert('RGB'))
    except Exception:
        return np.array(raw_img)
