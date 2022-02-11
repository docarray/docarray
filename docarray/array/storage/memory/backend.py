import itertools
from typing import (
    Generator,
    Iterator,
    Dict,
    Sequence,
    Optional,
    TYPE_CHECKING,
)


from .helper import _get_docs_ids, DocumentSeries
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
        from ... import DocumentArray

        self._data: DocumentSeries = DocumentSeries()
        self._id_to_index = {}
        if _docs is None:
            return
        elif isinstance(
            _docs,
            (
                DocumentArray,
                Sequence,
                Generator,
                Iterator,
                itertools.chain,
                DocumentSeries,
            ),
        ):
            if copy:
                _docs, ids = _get_docs_ids(_docs, copy=True)
                self._data = DocumentSeries(_docs, index=ids)
            elif isinstance(_docs, DocumentArray):
                self._data = _docs._data
            else:
                _docs, ids = _get_docs_ids(_docs)
                self._data = DocumentSeries(_docs, index=ids)

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
