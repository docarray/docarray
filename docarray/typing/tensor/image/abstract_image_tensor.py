import io
from abc import ABC

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class AbstractImageTensor(AbstractTensor, ABC):
    def to_bytes(self, format: str = 'PNG') -> bytes:
        """
        Convert image tensor to bytes.

        :param format: the image format use to store the image, can be 'PNG' , 'JPG' ...
        :return: bytes
        """
        from PIL import Image

        if format == 'jpg':
            format = 'jpeg'  # unify it to ISO standard

        tensor = self.get_comp_backend().to_numpy(self)

        mode = 'RGB' if tensor.ndim == 3 else 'L'
        pil_image = Image.fromarray(tensor, mode=mode)

        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=format)
            img_byte_arr = buffer.getvalue()

        return img_byte_arr

    def display(self) -> None:
        """
        Display image data from tensor.
        """
        from hubble.utils.notebook import is_notebook
        from PIL import Image

        np_array = self.get_comp_backend().to_numpy(self)
        img = Image.fromarray(np_array)

        if is_notebook():
            from IPython.display import display

            display(img)
        else:
            img.show()
