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
    Tuple,
    List,
)

from pandas import Series

from ..base.backend import BaseBackendMixin
from .... import Document

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


def _get_docs_ids(
    docs: Sequence['Document'], copy: bool = False
) -> Tuple[List['Document'], List[str]]:
    """ Returns a tuple of docs and ids while consuming the generator only once"""
    _docs, ids = [], []
    if copy:
        for doc in docs:
            _docs.append(Document(doc, copy=True))
            ids.append(doc.id)
    else:
        for doc in docs:
            _docs.append(Document(doc))
            ids.append(doc.id)
    return _docs, ids


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

        self._data: Series = Series()
        self._id_to_index = {}
        if _docs is None:
            return
        elif isinstance(
            _docs,
            (DocumentArray, Sequence, Generator, Iterator, itertools.chain, Series),
        ):
            if copy:
                _docs, ids = _get_docs_ids(_docs, copy=True)
                self._data = Series(_docs, index=ids)
            elif isinstance(_docs, DocumentArray):
                self._data = _docs._data
            else:
                _docs, ids = _get_docs_ids(_docs)
                self._data = Series(_docs, index=ids)

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
