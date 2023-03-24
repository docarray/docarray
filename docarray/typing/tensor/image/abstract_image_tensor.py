import io
import warnings
from abc import ABC
from typing import TYPE_CHECKING

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library, is_notebook


class AbstractImageTensor(AbstractTensor, ABC):
    def to_bytes(self, format: str = 'PNG') -> bytes:
        """
        Convert image tensor to bytes.

        :param format: the image format use to store the image, can be 'PNG' , 'JPG' ...
        :return: bytes
        """
        if TYPE_CHECKING:
            from PIL import Image as PILImage
        else:
            PILImage = import_library('PIL.Image')

        if format == 'jpg':
            format = 'jpeg'  # unify it to ISO standard

        tensor = self.get_comp_backend().to_numpy(self)

        mode = 'RGB' if tensor.ndim == 3 else 'L'
        pil_image = PILImage.fromarray(tensor, mode=mode)

        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=format)
            img_byte_arr = buffer.getvalue()

        return img_byte_arr

    def display(self) -> None:
        """
        Display image data from tensor in notebook.
        """
        if is_notebook():
            if TYPE_CHECKING:
                from PIL import Image as PILImage
            else:
                PILImage = import_library('PIL.Image')

            np_array = self.get_comp_backend().to_numpy(self)
            img = PILImage.fromarray(np_array)

            from IPython.display import display

            display(img)
        else:
            warnings.warn('Display of image is only possible in a notebook.')
