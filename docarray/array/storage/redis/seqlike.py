from typing import Iterable, Union

from docarray import Document, DocumentArray
from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods for DocumentArray with Redis as storage"""

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DA are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._client.client_info() == other._client.client_info()
            and self._config == other._config
        )

    def __len__(self):
        """Return the length of :class:`DocumentArray` that uses Redis as storage

        :return: the length of this :class:`DocumentArrayRedis` object
        """
        if self._list_like:
            return len(self._offset2ids)
        try:
            lua_script = f'return #redis.pcall("keys", "{self._config.index_name}:*")'
            cmd = self._client.register_script(lua_script)
            return cmd()
        except:
            return 0

    def __contains__(self, x: Union[str, 'Document']):
        """Check if ``x`` is contained in this :class:`DocumentArray` with Redis storage

        :param x: the id of the document to check or the document object itself
        :return: True if ``x`` is contained in self
        """
        if isinstance(x, str):
            return self._doc_id_exists(x)
        elif isinstance(x, Document):
            return self._doc_id_exists(x.id)
        else:
            return False

    def __repr__(self):
        """Return the string representation of :class:`DocumentArrayRedis` object
        :return: string representation of this object
        """
        return f'<DocumentArray[Redis] (length={len(self)}) at {id(self)}>'

    def _upload_batch(self, batch_of_docs: Iterable['Document']):
        pipe = self._client.pipeline()
        for doc in batch_of_docs:
            payload = self._document_to_redis(doc)
            pipe.hset(self._doc_prefix + doc.id, mapping=payload)
        pipe.execute()

    def _extend(self, docs: Iterable['Document']):
        da = DocumentArray(docs)
        for batch_of_docs in da.batch(self._config.batch_size):
            self._upload_batch(batch_of_docs)
            if self._list_like:
                self._offset2ids.extend(batch_of_docs[:, 'id'])
