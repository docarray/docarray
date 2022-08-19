from typing import Iterable, Union

from .... import Document
from ..base.seqlike import BaseSequenceLikeMixin


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
        try:
            # TODO
            # method 1
            # keys = self._client.keys(pattern) and add same prefix to all docs in one docarray
            # if self._offset2id_key.encode() in keys:
            #     return len(keys) - 1
            # else:
            #     return len(keys)

            # method 2
            # this way, extend(), insert() funcs have to call self._save_offset2ids()
            # if self._client.exists(self._offset2id_key.encode()):
            #     print('offset2id exists')
            #     return self._client.llen(self._offset2id_key.encode())
            # else:
            #     return 0

            # method 3
            return len(self._offset2ids)
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

    # TODO this del is unreachable, del will call __del__ in base/getsetdel
    # def __del__(self):
    #     """Delete this :class:`DocumentArrayRedis` object"""
    #     self._offset2ids.clear()

    def __repr__(self):
        """Return the string representation of :class:`DocumentArrayRedis` object
        :return: string representation of this object
        """
        return f'<DocumentArray[Redis] (length={len(self)}) at {id(self)}>'

    def _upload_batch(self, docs: Iterable['Document']):
        pipe = self._client.pipeline()
        batch = 0
        for doc in docs:
            payload = self._document_to_redis(doc)
            pipe.hset(self._doc_prefix + doc.id, mapping=payload)
            batch += 1
            if batch >= self._config.batch_size:
                pipe.execute()
                batch = 0
        if batch > 0:
            pipe.execute()

    def _extend(self, docs: Iterable['Document']):
        docs = list(docs)
        self._upload_batch(docs)
        self._offset2ids.extend([doc.id for doc in docs])
