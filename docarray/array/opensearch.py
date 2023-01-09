from docarray.array.document import DocumentArray
from docarray.array.storage.opensearch import StorageMixins, OpenSearchConfig

__all__ = ['DocumentArrayOpenSearch', 'OpenSearchConfig']


class DocumentArrayOpenSearch(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses OpenSearch as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayOpenSearch`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayOpenSearch` object
        """
        return super().__new__(cls)
