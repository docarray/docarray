import numpy as np

from docarray.typing import Tensor

from .any_url import AnyUrl


class ImageUrl(AnyUrl):
    def load(self) -> Tensor:
        """
        transform the url in a image Tensor

        this is just a patch we will move the function from old docarray
        :return: tensor image
        """

        return np.zeros((3, 224, 224))
