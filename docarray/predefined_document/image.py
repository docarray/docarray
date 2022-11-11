from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import Embedding, ImageUrl, Tensor


class Image(BaseDocument):
    """
    base Document for Image handling
    """

    uri: Optional[ImageUrl]
    tensor: Optional[Tensor]
    embedding: Optional[Embedding]
