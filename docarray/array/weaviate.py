from docarray.array.document import DocumentArray
from docarray.array.storage.weaviate import StorageMixins, WeaviateConfig

__all__ = ['DocumentArrayWeaviate', 'WeaviateConfig']


class DocumentArrayWeaviate(StorageMixins, DocumentArray):
    """
    DocumentArray that stores Documents in a `Weaviate <https://weaviate.io/>`_ vector search engine.

    .. note::
        This DocumentArray requires `weaviate-client`. You can install it via `pip install "docarray[weaviate]"`.

        To use Weaviate as storage backend, a Weaviate service needs to be running on your machine.

    With this implementation, :meth:`match` and :meth:`find` perform fast (approximate) vector search.
    Additionally, search with filters is supported.

    Example usage:

    .. code-block:: python

        from docarray import DocumentArray

        # connect to running Weaviate service with default configuration (address: http://localhost:8080)
        da = DocumentArray(storage='weaviate')

        # connect to a previously persisted DocumentArrayWeaviate by specifying name, host, and port
        da = DocumentArray(
            storage='weaviate', config={'name': 'Persisted', 'host': 'localhost', 'port': 1234}
        )


    .. seealso::
        For further details, see our :ref:`user guide <weaviate>`.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayWeaviate`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayWeaviate` object
        """
        return super().__new__(cls)
