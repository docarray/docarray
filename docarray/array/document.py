from typing import Optional, overload, TYPE_CHECKING, Dict, Union

from .base import BaseDocumentArray
from .mixins import AllMixins

if TYPE_CHECKING:
    from ..typing import DocumentArraySourceType
    from .memory import DocumentArrayInMemory
    from .sqlite import DocumentArraySqlite
    from .annlite import DocumentArrayAnnlite
    from .weaviate import DocumentArrayWeaviate
    from .elastic import DocumentArrayElastic
    from .storage.sqlite import SqliteConfig
    from .storage.annlite import AnnliteConfig
    from .storage.weaviate import WeaviateConfig
    from .storage.elastic import ElasticConfig


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
        storage: str = 'annlite',
        config: Optional[Union['AnnliteConfig', Dict]] = None,
    ) -> 'DocumentArrayAnnlite':
        """Create a AnnLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'elasticsearch',
        config: Optional[Union['ElasticConfig', Dict]] = None,
    ) -> 'DocumentArrayElastic':
        """Create a Elastic-powered DocumentArray object."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Ensures that offset2ids are stored in the db after
        operations in the DocumentArray are performed.
        """
        self._save_offset2ids()

    def __new__(cls, *args, storage: str = 'memory', **kwargs):
        if cls is DocumentArray:
            if storage == 'memory':
                from .memory import DocumentArrayInMemory

                instance = super().__new__(DocumentArrayInMemory)
            elif storage == 'sqlite':
                from .sqlite import DocumentArraySqlite

                instance = super().__new__(DocumentArraySqlite)
            elif storage == 'annlite':
                from .annlite import DocumentArrayAnnlite

                instance = super().__new__(DocumentArrayAnnlite)
            elif storage == 'weaviate':
                from .weaviate import DocumentArrayWeaviate

                instance = super().__new__(DocumentArrayWeaviate)
            elif storage == 'qdrant':
                from .qdrant import DocumentArrayQdrant

                instance = super().__new__(DocumentArrayQdrant)
            elif storage == 'elasticsearch':
                from .elastic import DocumentArrayElastic

                instance = super().__new__(DocumentArrayElastic)

            else:
                raise ValueError(f'storage=`{storage}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance
