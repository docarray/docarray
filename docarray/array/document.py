from typing import Optional, overload, TYPE_CHECKING, Dict, Union

from .base import BaseDocumentArray
from .mixins import AllMixins

if TYPE_CHECKING:
    from ..types import DocumentArraySourceType
    from .memory import DocumentArrayInMemory
    from .sqlite import DocumentArraySqlite
    from .pqlite import DocumentArrayPqlite
    from .weaviate import DocumentArrayWeaviate
    from .storage.sqlite import SqliteConfig
    from .storage.pqlite import PqliteConfig
    from .storage.weaviate import WeaviateConfig


class DocumentArray(AllMixins, BaseDocumentArray):
    @overload
    def __new__(
        cls, _docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ) -> 'DocumentArrayInMemory':
        """Create an in-memory DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'sqlite',
        config: Optional[Union['SqliteConfig', Dict]] = None,
    ) -> 'DocumentArraySqlite':
        """Create a SQLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'weaviate',
        config: Optional[Union['WeaviateConfig', Dict]] = None,
    ) -> 'DocumentArrayWeaviate':
        """Create a Weaviate-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'pqlite',
        config: Optional[Union['PqliteConfig', Dict]] = None,
    ) -> 'DocumentArrayPqlite':
        """Create a PQLite-powered DocumentArray object."""
        ...

    def __new__(cls, *args, storage: str = 'memory', **kwargs):
        if cls is DocumentArray:
            if storage == 'memory':
                from .memory import DocumentArrayInMemory

                instance = super().__new__(DocumentArrayInMemory)
            elif storage == 'sqlite':
                from .sqlite import DocumentArraySqlite

                instance = super().__new__(DocumentArraySqlite)
            elif storage == 'pqlite':
                from .pqlite import DocumentArrayPqlite

                instance = super().__new__(DocumentArrayPqlite)
            elif storage == 'weaviate':
                from .weaviate import DocumentArrayWeaviate

                instance = super().__new__(DocumentArrayWeaviate)
            else:
                raise ValueError(f'storage=`{storage}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance
