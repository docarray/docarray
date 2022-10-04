from typing import Optional, TYPE_CHECKING, Union, Dict
from dataclasses import dataclass

from docarray.array.storage.base.backend import BaseBackendMixin

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )


@dataclass
class MilvusConfig:
    config1: str
    config2: str
    config3: Dict
    ...


class BackendMixin(BaseBackendMixin):
    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[MilvusConfig, Dict]] = None,
        **kwargs
    ):
        super()._init_storage(_docs, config, **kwargs)
        ...

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        ...  # ensure unique identifiers here
        return config_joined
