import warnings
from abc import abstractmethod
from typing import Iterable, Iterator, MutableSequence

from docarray import Document, DocumentArray


class BaseSequenceLikeMixin(MutableSequence[Document]):
    """Implement sequence-like methods"""

    def _update_subindices_append_extend(self, value):
        if getattr(self, '_subindices', None):
            for selector, da in self._subindices.items():

                value = DocumentArray(value)

                if getattr(da, '_config', None) and da._config.root_id:
                    for v in value:
                        for doc in DocumentArray(v)[selector]:
                            doc.tags['_root_id_'] = v.id

                docs_selector = value[selector]
                if len(docs_selector) > 0:
                    da.extend(docs_selector)

    def insert(self, index: int, value: 'Document', **kwargs):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        :param kwargs: Additional Arguments that are passed to the Document Store. This has no effect for in-memory DocumentArray.
        """
        self._set_doc_by_id(value.id, value, **kwargs)
        self._offset2ids.insert(index, value.id)

    def append(self, value: 'Document', **kwargs):
        """Append `doc` to the end of the array.

        :param value: The doc needs to be appended.
        """
        self._append(value, **kwargs)
        self._update_subindices_append_extend(value)

    def _append(self, value, **kwargs):
        self._set_doc_by_id(value.id, value)
        self._offset2ids.append(value.id)

    @abstractmethod
    def __eq__(self, other):
        ...

    def __len__(self):
        ...

    def __iter__(self) -> Iterator['Document']:
        for _id in self._offset2ids:
            yield self._get_doc_by_id(_id)

    @abstractmethod
    def __contains__(self, other):
        ...

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._del_all_docs()

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def extend(self, values: Iterable['Document'], **kwargs) -> None:

        from docarray.helper import check_root_id

        if getattr(self, '_is_subindex', None):
            check_root_id(self, values)

        self._extend(values, **kwargs)
        self._update_subindices_append_extend(values)

    def _extend(self, values, **kwargs):
        for value in values:
            self._append(value, **kwargs)
