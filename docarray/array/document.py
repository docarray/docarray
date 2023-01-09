from contextlib import ExitStack
from typing import Optional, overload, TYPE_CHECKING, Dict, Union

from docarray.array.base import BaseDocumentArray
from docarray.array.mixins import AllMixins

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import DocumentArraySourceType
    from docarray.array.memory import DocumentArrayInMemory
    from docarray.array.sqlite import DocumentArraySqlite
    from docarray.array.annlite import DocumentArrayAnnlite
    from docarray.array.weaviate import DocumentArrayWeaviate
    from docarray.array.elastic import DocumentArrayElastic
    from docarray.array.redis import DocumentArrayRedis
    from docarray.array.milvus import DocumentArrayMilvus
    from docarray.array.opensearch import DocumentArrayOpenSearch
    from docarray.array.storage.sqlite import SqliteConfig
    from docarray.array.storage.annlite import AnnliteConfig
    from docarray.array.storage.weaviate import WeaviateConfig
    from docarray.array.storage.elastic import ElasticConfig
    from docarray.array.storage.redis import RedisConfig
    from docarray.array.storage.milvus import MilvusConfig
    from docarray.array.storage.opensearch import OpenSearchConfig


class DocumentArray(AllMixins, BaseDocumentArray):
    """
    DocumentArray is a list-like container of :class:`~docarray.Document` objects.

    A DocumentArray can be used to store, embed, and retrieve :class:`~docarray.Document` objects.

    .. code-block:: python

        from docarray import Document, DocumentArray

        da = DocumentArray(
            [Document(text='The cake is a lie'), Document(text='Do a barrel roll!')]
        )
        da.apply(Document.embed_feature_hashing)

        query = Document(text='Can i have some cake?').embed_feature_hashing()
        query.match(da, metric='jaccard', use_scipy=True)

        print(query.matches[:, ('text', 'scores__jaccard__value')])

    .. code-block:: bash

        [['The cake is a lie', 'Do a barrel roll!'], [0.9, 1.0]]

    A DocumentArray can also :ref:`embed its contents using a neural network <embed-via-model>`,
    process them using an :ref:`external Flow or Executor <da-post>`, and persist Documents in a :ref:`Document Store <doc-store>` for
    fast vector search:

    .. code-block:: python

        from docarray import Document, DocumentArray
        import numpy as np

        n_dim = 3
        metric = 'Euclidean'

        # initialize a DocumentArray with ANNLiter Document Store
        da = DocumentArray(
            storage='annlite',
            config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
        )
        # add Documents to the DocumentArray
        with da:
            da.extend(
                [
                    Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
                    for i in range(10)
                ]
            )
        # perform vector search
        np_query = np.ones(n_dim) * 8
        results = da.find(np_query)

    .. seealso::
        For further details, see our :ref:`user guide <documentarray>`.
    """

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        copy: bool = False,
        subindex_configs: Optional[Dict[str, 'None']] = None,
    ) -> 'DocumentArrayInMemory':
        """Create an in-memory DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'sqlite',
        config: Optional[Union['SqliteConfig', Dict]] = None,
        subindex_configs: Optional[Dict[str, Dict]] = None,
    ) -> 'DocumentArraySqlite':
        """Create a SQLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'weaviate',
        config: Optional[Union['WeaviateConfig', Dict]] = None,
        subindex_configs: Optional[Dict[str, Dict]] = None,
    ) -> 'DocumentArrayWeaviate':
        """Create a Weaviate-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'annlite',
        config: Optional[Union['AnnliteConfig', Dict]] = None,
        subindex_configs: Optional[Dict[str, Dict]] = None,
    ) -> 'DocumentArrayAnnlite':
        """Create a AnnLite-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'elasticsearch',
        config: Optional[Union['ElasticConfig', Dict]] = None,
        subindex_configs: Optional[Dict[str, Dict]] = None,
    ) -> 'DocumentArrayElastic':
        """Create a Elastic-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'redis',
        config: Optional[Union['RedisConfig', Dict]] = None,
    ) -> 'DocumentArrayRedis':
        """Create a Redis-powered DocumentArray object."""
        ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'milvus',
        config: Optional[Union['MilvusConfig', Dict]] = None,
    ) -> 'DocumentArrayMilvus':
        """Create a Milvus-powered DocumentArray object."""

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'opensearch',
        config: Optional[Union['OpenSearchConfig', Dict]] = None,
    ) -> 'DocumentArrayOpenSearch':
        """Create an OpenSearch-powered DocumentArray object."""
        ...

    def __enter__(self):
        self._exit_stack = ExitStack()
        # Ensure that we sync the data to the storage backend when exiting the context manager
        self._exit_stack.callback(self.sync)
        # Enter (and then exit) context of all subindices
        if getattr(self, '_subindices', None):
            for selector, da in self._subindices.items():
                self._exit_stack.enter_context(da)
        return self

    def __exit__(self, *args, **kwargs):
        # Trigger all __exit__()s and callbacks added in self.__enter__()
        self._exit_stack.close()

    def __new__(cls, *args, storage: str = 'memory', **kwargs):
        if cls is DocumentArray:
            if storage == 'memory':
                from docarray.array.memory import DocumentArrayInMemory

                instance = super().__new__(DocumentArrayInMemory)
            elif storage == 'sqlite':
                from docarray.array.sqlite import DocumentArraySqlite

                instance = super().__new__(DocumentArraySqlite)
            elif storage == 'annlite':
                from docarray.array.annlite import DocumentArrayAnnlite

                instance = super().__new__(DocumentArrayAnnlite)
            elif storage == 'weaviate':
                from docarray.array.weaviate import DocumentArrayWeaviate

                instance = super().__new__(DocumentArrayWeaviate)
            elif storage == 'qdrant':
                from docarray.array.qdrant import DocumentArrayQdrant

                instance = super().__new__(DocumentArrayQdrant)
            elif storage == 'elasticsearch':
                from docarray.array.elastic import DocumentArrayElastic

                instance = super().__new__(DocumentArrayElastic)
            elif storage == 'redis':
                from .redis import DocumentArrayRedis

                instance = super().__new__(DocumentArrayRedis)
            elif storage == 'milvus':
                from .milvus import DocumentArrayMilvus

                instance = super().__new__(DocumentArrayMilvus)
            elif storage == 'opensearch':
                from docarray.array.opensearch import DocumentArrayOpenSearch

                instance = super().__new__(DocumentArrayOpenSearch)

            else:
                raise ValueError(f'storage=`{storage}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance
