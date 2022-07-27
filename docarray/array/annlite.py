from docarray.array.document import DocumentArray
from docarray.array.storage.annlite import StorageMixins, AnnliteConfig

__all__ = ['AnnliteConfig', 'DocumentArrayAnnlite']


class DocumentArrayAnnlite(StorageMixins, DocumentArray):
    """
    DocumentArray that stores Documents in `ANNLite <https://github.com/jina-ai/annlite>`_.

    .. note::
        This DocumentArray requires `annlite`. You can install it via `pip install "docarray[annlite]"`.

    With this implementation, :meth:`match` and :meth:`find` perform fast (approximate) vector search.
    Additionally, search with filters on a :class:`~docarray.document.Document` s :attr:`~docarray.document.Document.tags` attribute is supported.

    Example usage:

    .. code-block:: python

        from docarray import Document, DocumentArray
        import numpy as np

        da = DocumentArray(storage='annlite', config={'data_path': './data', 'n_dim': 10})

        n_dim = 3
        da = DocumentArray(
            storage='annlite',
            config={
                'n_dim': n_dim,
                'columns': [('price', 'float')],
            },
        )

        with da:
            da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

        max_price = 3
        n_limit = 4

        filter = {'price': {'$lte': max_price}}
        results = da.find(filter=filter)

    .. seealso::
        For further details, see our :ref:`user guide <annlite>`.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
