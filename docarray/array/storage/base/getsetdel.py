from abc import abstractmethod, ABC
from typing import (
    Sequence,
    Any,
    Iterable,
)

from .... import Document


class BaseGetSetDelMixin(ABC):
    """Provide abstract methods and derived methods for ``__getitem__``, ``__setitem__`` and ``__delitem__``

    .. note::
        The following methods must be implemented:
            - :meth:`._get_doc_by_offset`
            - :meth:`._get_doc_by_id`
            - :meth:`._set_doc_by_offset`
            - :meth:`._set_doc_by_id`
            - :meth:`._del_doc_by_offset`
            - :meth:`._del_doc_by_id`

        Other methods implemented a generic-but-slow version that leverage the methods above.
        Please override those methods in the subclass whenever a more efficient implementation is available.
    """

    # Getitem APIs

    @abstractmethod
    def _get_doc_by_offset(self, offset: int) -> 'Document':
        ...

    @abstractmethod
    def _get_doc_by_id(self, _id: str) -> 'Document':
        ...

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_offset(o) for o in range(len(self))[_slice])

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_offset(o) for o in offsets)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_id`

        Override this function if there is a more efficient logic"""
        return (self._get_doc_by_id(_id) for _id in ids)

    # Delitem APIs

    @abstractmethod
    def _del_doc_by_offset(self, offset: int):
        ...

    @abstractmethod
    def _del_doc_by_id(self, _id: str):
        ...

    def _del_docs_by_slice(self, _slice: slice):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic"""
        for j in range(len(self))[_slice]:
            self._del_doc_by_offset(j)

    def _del_docs_by_mask(self, mask: Sequence[bool]):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic"""
        for idx, m in enumerate(mask):
            if not m:
                self._del_doc_by_offset(idx)

    def _del_all_docs(self):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic"""
        for j in range(len(self)):
            self._del_doc_by_offset(j)

    # Setitem API

    @abstractmethod
    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        ...

    @abstractmethod
    def _set_doc_by_id(self, _id: str, value: 'Document'):
        ...

    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        """
        if not isinstance(value, Iterable):
            raise TypeError(
                f'You right-hand assignment must be an iterable, receiving {type(value)}'
            )
        for _offset, val in zip(range(len(self))[_slice], value):
            self._set_doc_by_offset(_offset, val)

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Iterable['Document']
    ):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        """
        for _d, _v in zip(docs, values):
            _d._data = _v._data

        for _d in docs:
            if _d not in docs:
                root_d = self._find_root_doc(_d)
            else:
                # _d is already on the root-level
                root_d = _d

            if root_d:
                self._set_doc_by_id(root_d.id, root_d)

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        """
        d = self._get_doc_by_offset(offset)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc_by_offset(offset, d)

    def _set_doc_attr_by_id(self, _id: str, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        """
        d = self._get_doc_by_id(_id)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc_by_id(d.id, d)

    def _find_root_doc(self, d: Document):
        """Find `d`'s root Document in an exhaustive manner"""
        from docarray import DocumentArray

        for _d in self:
            _all_ids = set(DocumentArray(d)[...][:, 'id'])
            if d.id in _all_ids:
                return _d
