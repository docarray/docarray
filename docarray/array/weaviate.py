from .document import DocumentArray
from .storage.weaviate import StorageMixins, WeaviateConfig

__all__ = ['DocumentArrayWeaviate', 'WeaviateConfig']


class DocumentArrayWeaviate(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses Weaviate as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayWeaviate`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayWeaviate` object
        """
        return super().__new__(cls)
