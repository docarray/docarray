from typing import Union, Iterable, Dict
import warnings

from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin
from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods for DocumentArray with Elastic as storage"""

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._client.get_meta() == other._client.get_meta()
            and self._config == other._config
        )

    def __len__(self):
        """Return the length of :class:`DocumentArray` that uses Elastic as storage

        :return: the length of this :class:`DocumentArrayElastic` object
        """
        try:
            return self._client.count(index=self._config.index_name)["count"]
        except:
            return 0

    def __contains__(self, x: Union[str, 'Document']):
        """Check if ``x`` is contained in this :class:`DocumentArray` with Elastic storage

        :param x: the id of the document to check or the document object itself
        :return: True if ``x`` is contained in self
        """
        if isinstance(x, str):
            return self._doc_id_exists(x)
        elif isinstance(x, Document):
            return self._doc_id_exists(x.id)
        else:
            return False

    def __del__(self):
        """Delete this :class:`DocumentArrayElastic` object"""
        self._save_offset2ids()

        # if not self._persist:
        #    self._offset2ids.clear()

    def __repr__(self):
        """Return the string representation of :class:`DocumentArrayElastic` object
        :return: string representation of this object
        """
        return f'<{self.__class__.__name__} (length={len(self)}) at {id(self)}>'

    def _upload_batch(self, docs: Iterable['Document']):
        batch = []
        failed_index = []
        for doc in docs:
            batch.append(self._document_to_elastic(doc))
            if len(batch) > self._config.batch_size:
                failed_index.extend(self._send_requests(batch))
                self._refresh(self._config.index_name)
                batch = []
        if len(batch) > 0:
            failed_index.extend(self._send_requests(batch))
            self._refresh(self._config.index_name)

        return failed_index

    def extend(self, docs: Iterable['Document']):
        docs = list(docs)
        failed_index = self._upload_batch(docs)
        failed_ids = [index['_id'] for index in failed_index]
        self._offset2ids.extend(
            [
                doc.id
                for doc in docs
                if (doc.id not in self._offset2ids.ids) and (doc.id not in failed_ids)
            ]
        )

        if len(failed_ids) > 0:
            err_msg = f'fail to add Documents with ids: {failed_ids}'
            warnings.warn(err_msg)
            raise IndexError(err_msg)
