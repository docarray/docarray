from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table


class BaseBackendMixin(ABC):
    @abstractmethod
    def _init_storage(self, *args, **kwargs):
        ...

    def _fill_storage_table(self, table: 'Table'):
        table.show_header = False
        table.add_row('Class', self.__class__.__name__)
