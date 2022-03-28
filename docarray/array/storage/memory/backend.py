from typing import (
    Optional,
    TYPE_CHECKING,
    Iterable,
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

        self._data = {}
        if _docs is None:
            return
        elif isinstance(
            _docs,
            Iterable,
        ):
            if copy:
                for doc in _docs:
                    self.append(Document(doc, copy=True))
            else:
                self.extend(_docs)
        else:
            if isinstance(_docs, Document):
                if copy:
                    self.append(Document(_docs, copy=True))
                else:
                    self.append(_docs)
