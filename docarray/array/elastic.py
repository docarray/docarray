from .document import DocumentArray
from .storage.elastic import StorageMixins, ElasticConfig

__all__ = ['DocumentArrayElastic', 'ElasticConfig']


class DocumentArrayElastic(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses Elasticsearch as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayElastic`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayElastic` object
        """
        return super().__new__(cls)
