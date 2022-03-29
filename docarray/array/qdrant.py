from .document import DocumentArray
from .storage.qdrant import StorageMixins, QdrantConfig

__all__ = ['DocumentArrayQdrant', 'QdrantConfig']


class DocumentArrayQdrant(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses Qdrant as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayQdrant`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayQdrant` object
        """
        return super().__new__(cls)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Ensures that offset2ids are stored in the db after
        operations in the DocumentArray are performed.
        """
        self._update_offset2ids_meta()
