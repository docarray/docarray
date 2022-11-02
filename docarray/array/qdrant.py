from docarray.array.document import DocumentArray
from docarray.array.storage.qdrant import StorageMixins, QdrantConfig

__all__ = ['DocumentArrayQdrant', 'QdrantConfig']


class DocumentArrayQdrant(StorageMixins, DocumentArray):
    """
    DocumentArray that stores Documents in a `Qdrant <https://qdrant.tech/>`_ vector search engine.

    .. note::
        This DocumentArray requires `qdrant-client`. You can install it via `pip install "docarray[qdrant]"`.

        To use Qdrant as storage backend, a Qdrant service needs to be running on your machine.

    With this implementation, :meth:`match` and :meth:`find` perform fast (approximate) vector search.
    Additionally, search with filters is supported.

    Example usage:

    .. code-block:: python

        from docarray import DocumentArray

        # connect to running Qdrant service with default configuration (address: http://localhost:6333)
        da = DocumentArray(storage='qdrant', config={'n_dim': 10})

        # connect to a previously persisted DocumentArrayQdrant by specifying collection_name, host, and port
        da = DocumentArray(
            storage='qdrant',
            config={
                'collection_name': 'persisted',
                'host': 'localhost',
                'port': '6333',
                'n_dim': 10,
            },
        )


    .. seealso::
        For further details, see our :ref:`user guide <qdrant>`.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayQdrant`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayQdrant` object
        """
        return super().__new__(cls)
