import io
import warnings
from abc import ABC

import numpy as np
from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library, is_notebook

if TYPE_CHECKING:
    from docarray.typing.bytes.image_bytes import ImageBytes


class AbstractImageTensor(AbstractTensor, ABC):
    def to_bytes(self, format: str = 'PNG') -> 'ImageBytes':
        """
        Convert image tensor to [`ImageBytes`][docarray.typing.ImageBytes].

        :param format: the image format use to store the image, can be 'PNG' , 'JPG' ...
        :return: an ImageBytes object
        """
        PIL = import_library('PIL', raise_error=True)  # noqa: F841
        from PIL import Image as PILImage

        if format == 'jpg':
            format = 'jpeg'  # unify it to ISO standard

        tensor = self.get_comp_backend().to_numpy(self)

        mode = 'RGB' if tensor.ndim == 3 else 'L'
        pil_image = PILImage.fromarray(tensor, mode=mode)

        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=format)
            img_byte_arr = buffer.getvalue()

        from docarray.typing.bytes.image_bytes import ImageBytes

        return ImageBytes(img_byte_arr)

    def save(self, file_path: str) -> None:
        """
        Save image tensor to an image file.

        :param file_path: path to an image file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        """
        PIL = import_library('PIL', raise_error=True)  # noqa: F841
        from PIL import Image as PILImage

        comp_backend = self.get_comp_backend()
        np_img = comp_backend.to_numpy(self).astype(np.uint8)

        pil_img = PILImage.fromarray(np_img)
        pil_img.save(file_path)

    def display(self) -> None:
        """
        Display image data from tensor in notebook.
        """
        if is_notebook():
            PIL = import_library('PIL', raise_error=True)  # noqa: F841
            from PIL import Image as PILImage

            np_array = self.get_comp_backend().to_numpy(self)
            img = PILImage.fromarray(np_array)

            from IPython.display import display

            display(img)
        else:
            warnings.warn('Display of image is only possible in a notebook.')
