import itertools
from typing import (
    Generator,
    Iterator,
    Dict,
    Sequence,
    Optional,
    TYPE_CHECKING,
)

from ..base.backend import BaseBackendMixin
from .... import Document

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


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

    def _init_storage(
        self, _docs: Optional["DocumentArraySourceType"] = None, copy: bool = False
    ):
        from ... import DocumentArray

        self._needs_id2offset_rebuild = True
        # self._id2offset needs to be rebuilt after every insert or delete
        # this flag allows to do it lazily and cache the result

        self._data = []
        if _docs is None:
            return
        elif isinstance(
            _docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            if copy:
                self._data = [Document(d, copy=True) for d in _docs]
            elif isinstance(_docs, DocumentArray):
                self._data = _docs._data
                self._id_to_index = _docs._id2offset
            else:
                self._data = list(_docs)
        else:
            if isinstance(_docs, Document):
                if copy:
                    self.append(Document(_docs, copy=True))
                else:
                    self.append(_docs)
