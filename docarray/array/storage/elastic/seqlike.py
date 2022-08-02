from typing import Union, Iterable, Dict, List
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

    @staticmethod
    def _parse_index_ids_from_bulk_info(
        accumulated_info: List[Dict],
    ) -> Dict[str, List[int]]:
        """Parse ids from bulk info of failed send request to ES operation

        :param accumulated_info: accumulated info of failed operation
        :return: dict containing failed index ids of each operation type
        """

        parsed_ids = {}

        for info in accumulated_info:
            for _op_type in info.keys():
                if '_id' in info[_op_type]:
                    if _op_type not in parsed_ids:
                        parsed_ids[_op_type] = []

                    parsed_ids[_op_type].append(info[_op_type]['_id'])

        return parsed_ids

    def _upload_batch(self, docs: Iterable['Document']) -> List[int]:
        batch = []
        accumulated_info = []
        for doc in docs:
            batch.append(self._document_to_elastic(doc))
            if len(batch) > self._config.batch_size:
                accumulated_info.extend(self._send_requests(batch))
                self._refresh(self._config.index_name)
                batch = []
        if len(batch) > 0:
            accumulated_info.extend(self._send_requests(batch))
            self._refresh(self._config.index_name)

        successful_ids = self._parse_index_ids_from_bulk_info(accumulated_info)
        if 'index' not in successful_ids:
            return []

        return successful_ids['index']

    def extend(self, docs: Iterable['Document']):
        docs = list(docs)
        successful_indexed_ids = self._upload_batch(docs)
        self._offset2ids.extend(
            [_id for _id in successful_indexed_ids if _id not in self._offset2ids.ids]
        )

        if len(successful_indexed_ids) != len(docs):
            doc_ids = [doc.id for doc in docs]
            failed_index_ids = set(doc_ids) - set(successful_indexed_ids)

            err_msg = f'fail to add Documents with ids: {failed_index_ids}'
            warnings.warn(err_msg)
            raise IndexError(err_msg)
