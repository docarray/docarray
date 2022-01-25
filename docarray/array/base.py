from abc import ABC


class BaseDocumentArray(ABC):
    def __init__(self, *args, storage: str, **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)
