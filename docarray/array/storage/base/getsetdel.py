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
        Override this function if there is a more efficient logic

        :param _slice: the slice used for indexing
        :return: an iterable of document
        """
        return (self._get_doc_by_offset(o) for o in range(len(self))[_slice])

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`
        Override this function if there is a more efficient logic

        :param offsets: the offsets used for indexing
        :return: an iterable of document
        """
        return (self._get_doc_by_offset(o) for o in offsets)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_id`
        Override this function if there is a more efficient logic

        :param ids: the ids used for indexing
        :return: an iterable of document
        """
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
        Override this function if there is a more efficient logic
        :param _slice: the slice used for indexing
        """
        for j in range(len(self))[_slice]:
            self._del_doc_by_offset(j)

    def _del_docs_by_mask(self, mask: Sequence[bool]):
        """This function is derived and may not have the most efficient implementation.
        Override this function if there is a more efficient logic
        :param mask: the boolean mask used for indexing
        """
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
        :param _slice: the slice used for indexing
        :param value: the value docs will be updated to
        :raises TypeError: error raised when right-hand assignment is not an iterable
        """
        if not isinstance(value, Iterable):
            raise TypeError(
                f'You right-hand assignment must be an iterable, receiving {type(value)}'
            )
        for _offset, val in zip(range(len(self))[_slice], value):
            self._set_doc_by_offset(_offset, val)

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Sequence['Document']
    ):
        docs = list(docs)
        if len(docs) != len(values):
            raise ValueError(
                f'length of docs to set({len(docs)}) does not match '
                f'length of values({len(values)})'
            )

        for _d, _v in zip(docs, values):
            self._set_doc_by_id(_d.id, _v)

    def _set_doc_value_pairs_nested(
        self, docs: Iterable['Document'], values: Sequence['Document']
    ):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        :param docs: the docs to update
        :param values: the value docs will be updated to
        """
        docs = list(docs)
        if len(docs) != len(values):
            raise ValueError(
                f'length of docs to set({len(docs)}) does not match '
                f'length of values({len(values)})'
            )

        for _d, _v in zip(docs, values):
            if _d.id != _v.id:
                raise ValueError(
                    'Setting Documents by traversal paths with different IDs is not supported'
                )
            _d._data = _v._data
            if _d not in self:
                root_d = self._find_root_doc_and_modify(_d)
            else:
                # _d is already on the root-level
                root_d = _d

            if root_d:
                self._set_doc_by_id(root_d.id, root_d)

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        :param offset: the offset used for indexing
        :param attr: the attribute of document to update
        :param value: the value doc's attr will be updated to
        """
        d = self._get_doc_by_offset(offset)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc_by_offset(offset, d)

    def _set_doc_attr_by_id(self, _id: str, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        :param _id: the id used for indexing
        :param attr: the attribute of document to update
        :param value: the value doc's attr will be updated to
        """
        d = self._get_doc_by_id(_id)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc_by_id(_id, d)

    def _find_root_doc_and_modify(self, d: Document) -> 'Document':
        """Find `d`'s root Document in an exhaustive manner
        :param: d: the input document
        :return: the root of the input document
        """
        from docarray import DocumentArray

        for _d in self:
            da = DocumentArray(_d)[...]
            _all_ids = set(da[:, 'id'])
            if d.id in _all_ids:
                da[d.id].copy_from(d)
                return _d
