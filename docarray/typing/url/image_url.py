import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.utils._internal.misc import is_notebook

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


T = TypeVar('T', bound='ImageUrl')

IMAGE_FILE_FORMATS = ('png', 'jpeg', 'jpg')


@_register_proto(proto_type_name='image_url')
class ImageUrl(AnyUrl):
    """
    URL to a .png, .jpeg, or .jpg file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, str, Any],
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

    def load_pil(self, timeout: Optional[float] = None) -> 'PILImage.Image':
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
        from docarray.typing.bytes.image_bytes import ImageBytes

        return ImageBytes(self.load_bytes(timeout=timeout)).load_pil()

    def load(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis_layout: Tuple[str, str, str] = ('H', 'W', 'C'),
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray image tensor

        ---

        ```python
        from docarray import BaseDoc
        from docarray.typing import ImageUrl
        import numpy as np


        class MyDoc(BaseDoc):
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
        ```

        ---

        :param width: width of the image tensor.
        :param height: height of the image tensor.
        :param axis_layout: ordering of the different image axes.
            'H' = height, 'W' = width, 'C' = color channel
        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: np.ndarray representing the image as RGB values
        """
        from docarray.typing.bytes.image_bytes import ImageBytes

        buffer = ImageBytes(self.load_bytes(timeout=timeout))
        return buffer.load(width, height, axis_layout)

    def display(self) -> None:
        """
        Display image data from url in notebook.
        """
        if is_notebook():
            from IPython.display import Image, display

            remote_url = True if self.startswith('http') else False
            if remote_url:
                display(Image(url=self))
            else:
                display(Image(filename=self))
        else:
            warnings.warn('Display of image is only possible in a notebook.')
