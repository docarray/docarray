from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import is_dataclass, asdict
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from docarray.typing import DocumentArraySourceType, ArrayType

TypeMap = namedtuple('TypeMap', ['type', 'converter'])


class BaseBackendMixin(ABC):
    TYPE_MAP: Dict[str, TypeMap]

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        self._load_offset2ids()

    def _init_subindices(self, *args, **kwargs):
        self._subindices = {}
        subindex_configs = kwargs.get('subindex_configs', None)
        if subindex_configs:
            config = asdict(self._config) if getattr(self, '_config', None) else dict()

            for name, config_subindex in subindex_configs.items():
                config_subindex = (
                    dict() if config_subindex is None else config_subindex
                )  # allow None as input
                if is_dataclass(config_subindex):
                    config_subindex = asdict(config_subindex)
                config_joined = {**config, **config_subindex}
                config_joined = self._ensure_unique_config(
                    config, config_subindex, config_joined, name
                )
                self._subindices[name] = self.__class__(config=config_joined)
                self._subindices[name].extend(self.traverse_flat(name[1:]))

    @abstractmethod
    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        """
        Ensures that the subindex configuration is unique, despite it inheriting unpopulated fields from the root config.

        :param config_root: The configuration of the root index.
        :param config_subindex: The configuration that was explicitly provided by the user for the subindex.
        :param config_joined: The configuration that combines root and subindex configs. This is the configuration that will be used for subindex construction.
        :param subindex_name: Name (access path) of the subindex
        :return: config_joined that is unique compared to config_root
        """
        ...

    def _get_storage_infos(self) -> Optional[Dict]:
        if hasattr(self, '_config') and is_dataclass(self._config):
            return {k: str(v) for k, v in asdict(self._config).items()}

    def _map_id(self, _id: str) -> str:
        return _id

    def _map_column(self, value, col_type) -> str:
        return self.TYPE_MAP[col_type].converter(value)

    def _map_embedding(self, embedding: 'ArrayType') -> 'ArrayType':
        from docarray.math.ndarray import to_numpy_array

        return to_numpy_array(embedding)

    def _map_type(self, col_type: str) -> str:
        return self.TYPE_MAP[col_type].type

    def _normalize_columns(self, columns):
        if columns is None:
            return []
        return columns
