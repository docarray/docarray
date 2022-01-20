from abc import ABC


class BaseDocumentArray(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)
