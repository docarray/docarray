from abc import ABC, abstractmethod
from typing import Dict


class BaseBackendMixin(ABC):
    @abstractmethod
    def _init_storage(self, *args, **kwargs):
        ...

    def _get_storage_infos(self) -> Dict:
        return {'Class': self.__class__.__name__}
