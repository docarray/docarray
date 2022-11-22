from .document import DocumentArray

from .storage.milvus import StorageMixins, MilvusConfig

__all__ = ['MilvusConfig', 'DocumentArrayMilvus']


class DocumentArrayMilvus(StorageMixins, DocumentArray):
    """
    DocumentArray that stores Documents in a `Milvus <https://milvus.io//>`_ vector search engine.

    .. note::
        This DocumentArray requires `pymilvus`. You can install it via `pip install "docarray[milvus]"`.

        To use Milvus as storage backend, a Milvus service needs to be running on your machine.

    With this implementation, :meth:`match` and :meth:`find` perform fast (approximate) vector search.
    Additionally, search with filters is supported.

    Example usage:

    .. code-block:: python

        from docarray import DocumentArray

        # connect to running Milvus service with default configuration (address: http://localhost:19530)
        da = DocumentArray(storage='milvus', config={'n_dim': 10})

        # connect to a previously persisted DocumentArrayMilvus by specifying collection_name, host, and port
        da = DocumentArray(
            storage='milvus',
            config={
                'collection_name': 'persisted',
                'host': 'localhost',
                'port': '19530',
                'n_dim': 10,
            },
        )


    .. seealso::
        For further details, see our :ref:`user guide <milvus>`.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
