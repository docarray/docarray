from __future__ import annotations

from typing import Any, Dict, Optional

from docarray import BaseDocument, DocumentArray
from docarray.typing import AnyEmbedding, AnyTensor


class Document(BaseDocument):

    tensor: Optional[AnyTensor]
    chunks: Optional[DocumentArray[Document]]
    matches: Optional[DocumentArray[Document]]
    blob: Optional[Any]
    text: Optional[str]
    url: Optional[str]
    embedding: Optional[AnyEmbedding]
    # tags: Dict[str, Any] = dict()
    scores: Optional[Dict[str, Any]]
