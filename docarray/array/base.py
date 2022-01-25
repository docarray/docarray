from typing import MutableSequence

from .. import Document


class BaseDocumentArray(MutableSequence[Document]):
    def __init__(self, *args, storage: str = 'memory', **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)
