from abc import ABC, abstractmethod


class BaseBackendMixin(ABC):
    @abstractmethod
    def _init_storage(self, *args, **kwargs):
        ...

    @classmethod
    def _default_protocol(cls):
        return 'pickle-array'
