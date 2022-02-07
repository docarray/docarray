import functools
import itertools
from typing import (
    Generator,
    Iterator,
    Dict,
    Sequence,
    Optional,
    TYPE_CHECKING,
    Callable,
)

from ..base.backend import BaseBackendMixin
from .... import Document

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


def needs_id2offset_rebuild(func) -> Callable:
    # self._id2offset needs to be rebuilt after every insert or delete
    # this flag allows to do it lazily and cache the result
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._needs_id2offset_rebuild = True
        return func(self, *args, **kwargs)

    return wrapper


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    @property
    def _id2offset(self) -> Dict[str, int]:
        """Return the `_id_to_index` map

        :return: a Python dict.
        """
        if self._needs_id2offset_rebuild:
            self._rebuild_id2offset()

        return self._id_to_index

    def _rebuild_id2offset(self) -> None:
        """Update the id_to_index map by enumerating all Documents in self._data.

        Very costy! Only use this function when self._data is dramtically changed.
        """

        self._id_to_index = {
            d.id: i for i, d in enumerate(self._data)
        }  # type: Dict[str, int]

        self._needs_id2offset_rebuild = False

    @needs_id2offset_rebuild
    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        copy: bool = False,
        *args,
        **kwargs
    ):
        from ... import DocumentArray
        from ...memory import DocumentArrayInMemory

        self._data = []
        self._id_to_index = {}
        if _docs is None:
            return
        elif isinstance(
            _docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            if copy:
                self._data = [Document(d, copy=True) for d in _docs]
            elif isinstance(_docs, DocumentArray):
                self._data = _docs._data
            else:
                self._data = list(_docs)

            if isinstance(_docs, DocumentArrayInMemory):
                self._id_to_index = _docs._id2offset
                self._needs_id2offset_rebuild = _docs._needs_id2offset_rebuild
        else:
            if isinstance(_docs, Document):
                if copy:
                    self.append(Document(_docs, copy=True))
                else:
                    self.append(_docs)

    def _get_storage_infos(self) -> Dict:
        storage_infos = super()._get_storage_infos()
        storage_infos['Backend'] = 'In Memory'
        return storage_infos
