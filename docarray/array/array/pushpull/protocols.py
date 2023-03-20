from typing import Dict, Iterator, List, Optional, Type

from typing_extensions import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from docarray import BaseDocument, DocumentArray


class PushPullLike(Protocol):
    @staticmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        ...

    @staticmethod
    def delete(name: str, missing_ok: bool) -> bool:
        ...

    @staticmethod
    def push(
        da: 'DocumentArray',
        url: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        ...

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDocument'],
        url: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        ...

    @staticmethod
    def pull(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocumentArray':
        ...

    @staticmethod
    def pull_stream(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDocument']:
        ...
