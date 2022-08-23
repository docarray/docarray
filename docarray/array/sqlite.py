from docarray.array.document import DocumentArray

from docarray.array.storage.sqlite import StorageMixins, SqliteConfig

__all__ = ['SqliteConfig', 'DocumentArraySqlite']


class DocumentArraySqlite(StorageMixins, DocumentArray):
    """
    DocumentArray that stores Documents in a `SQLite database <https://www.sqlite.org/index.html>`_.
    This stores Documents on disk instead of keeping them in memory, and offers the simplest way of persisting data with DocArray.

    With this implementation, :meth:`match` and :meth:`find` perform exact (exhaustive) vector search.

    Example usage:

    .. code-block:: python

        from docarray import DocumentArray

        # with default config
        da = DocumentArray(storage='sqlite')

        # with customized config
        da1 = DocumentArray(storage='sqlite', config={'connection': 'example.db'})

        # connect to a previously created database
        da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'mine'}
        )


    .. seealso::
        For further details, see our :ref:`user guide <sqlite>`.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
