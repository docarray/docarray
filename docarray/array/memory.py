from docarray.array.document import DocumentArray
from docarray.array.storage.memory import StorageMixins


class DocumentArrayInMemory(StorageMixins, DocumentArray):
    """
    Default DocumentArray that stores Documents in memory.
    With this implementation, :meth:`match` and :meth:`find` perform exact (exhaustive) vector search.

    Example usage:

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

    .. seealso::
        For further details, see our :ref:`user guide <documentarray>`.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
