from typing import ClassVar, Dict

from typing_extensions import Protocol


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict]
