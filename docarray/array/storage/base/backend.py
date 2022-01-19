from abc import ABC, abstractmethod


class BaseBackendMixin(ABC):
    @abstractmethod
    def _init_storage(self, *args, **kwargs):
        ...
