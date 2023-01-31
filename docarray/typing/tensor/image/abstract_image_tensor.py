import io
from abc import ABC, abstractmethod

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class AbstractImageTensor(AbstractTensor, ABC):
    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Convert image tensor to bytes.
        """
        from PIL import Image

        tensor = self.get_comp_backend().to_numpy(self)
        pil_image = Image.fromarray(tensor)

        with io.BytesIO() as buffer:
            pil_image.save(buffer, format='PNG')
            img_byte_arr = buffer.getvalue()

        return img_byte_arr
