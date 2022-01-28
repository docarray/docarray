from typing import Optional, overload, TYPE_CHECKING, Dict, Union

from .base import BaseDocumentArray
from .mixins import AllMixins

if TYPE_CHECKING:
    from ..types import (
        DocumentArraySourceType,
        DocumentArrayLike,
        DocumentArraySqlite,
        DocumentArrayInMemory,
    )
    from .storage.sqlite import SqliteConfig


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

    def __new__(cls, *args, storage: str = 'memory', **kwargs) -> 'DocumentArrayLike':
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
