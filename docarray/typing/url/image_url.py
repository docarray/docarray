import io
import struct
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from docarray.proto import NodeProto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper import _uri_to_blob

if TYPE_CHECKING:
    import PIL

IMAGE_FILE_FORMATS = ('png', 'jpeg', 'jpg')


class ImageUrl(AnyUrl):
    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(image_url=str(self))

    # TODO(johannes) add validation for image URI

    def load(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        channel_axis: int = -1,
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray image tensor

        :param width: width of the image tensor.
        :param height: height of the image tensor.
        :param channel_axis: axis where to put the image color channel;
            ``-1`` indicates the color channel info at the last axis
        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: np.ndarray representing the image as RGB values
        """
        # TODO(johannes) the axis argument is confusing, because it if it is not -1,
        #  it is unclear where width and height end up
        # instead we should allow the user to pass an axis for width, height, and axis

        buffer = _uri_to_blob(self, timeout=timeout)
        tensor = _to_image_tensor(io.BytesIO(buffer), width=width, height=height)
        return _move_channel_axis(tensor, target_channel_axis=channel_axis)

    def load_to_bytes(
        self,
        image_format: str = 'png',
        width: Optional[int] = None,
        height: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Load image at URL to bytes (buffer).

        :param image_format: File format of the file located the the url.
            Supported formats are `png`, `jpg`, and `jpeg`.
        :param width: Before converting to bytes, resize the image to this width.
        :param height: Before converting to bytes, resize the image to this height.
        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: The image as bytes (buffer).
        """
        channel_axis = -1
        image_tensor = self.load(
            width=width, height=height, channel_axis=channel_axis, timeout=timeout
        )
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
    tensor: np.ndarray, original_channel_axis: int = -1, target_channel_axis: int = -1
) -> np.ndarray:
    """Moves channel axis around."""
    if original_channel_axis != target_channel_axis:
        tensor = np.moveaxis(tensor, original_channel_axis, target_channel_axis)
    return tensor


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
