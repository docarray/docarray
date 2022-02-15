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


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        copy: bool = False,
        *args,
        **kwargs
    ):
        super()._init_storage(_docs, copy=copy, *args, **kwargs)

        from ...memory import DocumentArrayInMemory

        self._data = {}
        if _docs is None:
            return
        elif isinstance(
            _docs,
            (DocumentArrayInMemory, Sequence, Generator, Iterator, itertools.chain),
        ):
            if copy:
                self._data = {d.id: Document(d, copy=True) for d in _docs}
            elif isinstance(_docs, DocumentArrayInMemory):
                self._data = _docs._data
            else:
                self._data = {doc.id: doc for doc in _docs}
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
