from typing import Union, Iterable

from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin
from docarray.array.memory import DocumentArrayInMemory
from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def extend(self, values: Iterable['Document']) -> None:
        docs = DocumentArrayInMemory(values)
        if len(docs) == 0:
            return

        for doc in docs:
            doc.embedding = self._map_embedding(doc.embedding)

        self._annlite.index(docs)
        self._offset2ids.extend([doc.id for doc in docs])

        self._update_subindices_append_extend(docs)

    def append(self, value: 'Document'):
        self.extend([value])

    def __eq__(self, other):
        """In annlite backend, data are considered as identical if configs point to the same database source"""
        return (
            type(self) is type(other)
            and type(self._config) is type(other._config)
            and self._config == other._config
        )

    def __repr__(self):
        return f'<DocumentArray[AnnLite] (length={len(self)}) at {id(self)}>'

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._annlite.get_doc_by_id(x) is not None
        elif isinstance(x, Document):
            return self._annlite.get_doc_by_id(x.id) is not None
        else:
            return False
