from typing import Iterable, Iterator, Union, TYPE_CHECKING
from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin
from docarray.array.storage.milvus.backend import _batch_list, _always_true_expr
from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other have the same Milvus Collections for data and offset2id

        :param other: the other object to check for equality
        :return: `True` if other is equal to self
        """
        return (
            type(self) is type(other)
            and self._collection.name == other._collection.name
            and self._offset2id_collection.name == other._offset2id_collection.name
            and self._config == other._config
        )

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, Document):
            x = x.id
        try:
            self._get_doc_by_id(x)
            return True
        except:
            return False

    def __repr__(self):
        return f'<DocumentArray[Milvus] (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Iterable['Document']]):
        if isinstance(other, Document):
            self.append(other)
        else:
            self.extend(other)
        return self

    def insert(self, index: int, value: 'Document', **kwargs):
        self._set_doc_by_id(value.id, value, **kwargs)
        self._offset2ids.insert(index, value.id)

    def _append(self, value: 'Document', **kwargs):
        self._set_doc_by_id(value.id, value, **kwargs)
        self._offset2ids.append(value.id)

    def _extend(self, values: Iterable['Document'], **kwargs):
        docs = list(values)
        if not docs:
            return
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        kwargs = self._update_kwargs_from_config('batch_size', **kwargs)
        for docs_batch in _batch_list(list(docs), kwargs['batch_size']):
            payload = self._docs_to_milvus_payload(docs_batch)
            self._collection.insert(payload, **kwargs)
            self._offset2ids.extend([doc.id for doc in docs_batch])

    def __len__(self):
        if self._list_like:
            return len(self._offset2ids)
        else:
            # Milvus has no native way to get num of entities
            # so only use it as fallback option
            with self.loaded_collection():
                res = self._collection.query(
                    expr=_always_true_expr('document_id'),
                    output_fields=['document_id'],
                )
                return len(res)
