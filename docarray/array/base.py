import atexit
from weakref import WeakMethod
from typing import MutableSequence, TYPE_CHECKING, Union, Iterable

from docarray import Document

if TYPE_CHECKING:
    from docarray.typing import T


class BaseDocumentArray(MutableSequence[Document]):
    def __init__(self, *args, storage: str = 'memory', **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)
        self._init_subindices(*args, **kwargs)
        self._register_exit_hook()

    def _register_exit_hook(self):
        weakref_del = WeakMethod(self.__del__)

        def exit_hook():
            if weakref_del() is not None:
                weakref_del()()

        atexit.register(exit_hook)
        self._deleted = False

    def __add__(self: 'T', other: Union['Document', Iterable['Document']]) -> 'T':
        v = type(self)(self)
        v.extend(other)
        return v
