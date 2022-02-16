from typing import Union, Iterable, Sequence

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document

from ...memory import DocumentArrayInMemory


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def extend(self, values: Iterable['Document']) -> None:
        docs = DocumentArrayInMemory(values)
        if len(docs) == 0:
            return

        for doc in docs:
            self._to_numpy_embedding(doc)

        self._pqlite.index(docs)
        self._offset2ids.extend([doc.id for doc in docs])

    def __del__(self) -> None:
        if not self._persist:
            self._offset2ids.clear()
            self._pqlite.clear()

    def __eq__(self, other):
        """In pqlite backend, data are considered as identical if configs point to the same database source"""
        return (
            type(self) is type(other)
            and type(self._config) is type(other._config)
            and self._config == other._config
        )

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def __repr__(self):
        return f'<DocumentArray[PQLite] (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Sequence['Document']]):
        v = type(self)(self)
        v.extend(other)
        return v

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return self._pqlite.get_doc_by_id(x) is not None
        elif isinstance(x, Document):
            return self._pqlite.get_doc_by_id(x.id) is not None
        else:
            return False
