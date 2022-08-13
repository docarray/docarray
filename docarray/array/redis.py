from .document import DocumentArray
from .storage.redis import RedisConfig, StorageMixins

__all__ = ['DocumentArrayRedis', 'RedisConfig']


class DocumentArrayRedis(StorageMixins, DocumentArray):
    """This is a :class:`DocumentArray` that uses Redis as
    vector search engine and storage.
    """

    def __new__(cls, *args, **kwargs):
        """``__new__`` method for :class:`DocumentArrayRedis`

        :param *args: list of args to instantiate the object
        :param **kwargs: dict of args to instantiate the object
        :return: the instantiated :class:`DocumentArrayRedis` object
        """
        return super().__new__(cls)
