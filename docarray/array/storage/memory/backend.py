import itertools
from typing import (
    Generator,
    Iterator,
    Dict,
    Sequence,
    Optional,
    TYPE_CHECKING,
)

from .... import Document
from ..base.backend import BaseBackendMixin

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


class MemoryBackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    @property
    def _id2offset(self) -> Dict[str, int]:
        """Return the `_id_to_index` map

        :return: a Python dict.
        """
        if not hasattr(self, '_id_to_index'):
            self._rebuild_id2offset()
        return self._id_to_index

    def _rebuild_id2offset(self) -> None:
        """Update the id_to_index map by enumerating all Documents in self._data.

        Very costy! Only use this function when self._data is dramtically changed.
        """

        self._id_to_index = {
            d.id: i for i, d in enumerate(self._data)
        }  # type: Dict[str, int]

    def _init_storage(
        self, docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ):
        from ... import DocumentArray

        self._data = []
        if docs is None:
            return
        elif isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            if copy:
                self._data = [Document(d, copy=True) for d in docs]
                self._rebuild_id2offset()
            elif isinstance(docs, DocumentArray):
                self._data = docs._data
                self._id_to_index = docs._id2offset
            else:
                self._data = list(docs)
                self._rebuild_id2offset()
        else:
            if isinstance(docs, Document):
                if copy:
                    self.append(Document(docs, copy=True))
                else:
                    self.append(docs)
