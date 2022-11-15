from typing import Optional

from docarray.document import BaseDocument
from docarray.typing.embedding import Embedding, Tensor


class Text(BaseDocument):
    """
    base Document for Text handling
    """

    text: str = ''
    tensor: Optional[Tensor]
    embedding: Optional[Embedding]
