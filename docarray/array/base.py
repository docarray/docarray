from typing import MutableSequence, TYPE_CHECKING, Union, Iterable, Type

from .. import Document

if TYPE_CHECKING:
    from ..types import T


class BaseDocumentArray(MutableSequence[Document]):
    def __init__(self, *args, storage: str = 'memory', **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)

    def __add__(self: Type['T'], other: Union['Document', Iterable['Document']]) -> 'T':
        v = type(self)(self)
        v.extend(other)
        return v
