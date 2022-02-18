from abc import ABC
from typing import Dict, Optional, TYPE_CHECKING

from docarray.array.storage.base.helper import Offset2ID

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


class BaseBackendMixin(ABC):
    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        copy: bool = False,
        *args,
        **kwargs
    ):
        self._load_offset2ids()

    def _get_storage_infos(self) -> Dict:
        return {'Class': self.__class__.__name__}
