import itertools
from abc import abstractmethod, ABC
from typing import (
    Sequence,
    Any,
    Iterable,
    Dict,
)

from .helper import Offset2ID
from .... import Document


class BaseGetSetDelMixin(ABC):
    """Provide abstract methods and derived methods for ``__getitem__``, ``__setitem__`` and ``__delitem__``

    .. note::
        The following methods must be implemented:
            - :meth:`._get_doc_by_id`
            - :meth:`._set_doc_by_id`
            - :meth:`._del_doc_by_id`
        Keep in mind that these methods above ** must not ** handle offset2id of the DocumentArray.

        These methods are actually wrapped by the following methods which handle the offset2id:
            - :meth:`._set_doc`
            - :meth:`._del_doc`
            - :meth:`._del_all_docs`

        Therefore, you should make sure to use the wrapper methods in case you expect offset2id to be updated, and use
        the inner methods in case you don't want to handle offset2id (for example, if you want to handle it in a
        later step)

        Other methods implemented a generic-but-slow version that leverage the methods above.
        Please override those methods in the subclass whenever a more efficient implementation is available.
        Mainly, if the backend storage supports operations in batches, you can implement the following methods:
            - :meth:`._get_docs_by_ids`
            - :meth:`._set_docs_by_ids`
            - :meth:`._del_docs_by_ids`
            - :meth:`._clear_storage`

        Likewise, the methods above do not handle offset2id. They are wrapped by the following methods that update the
        offset2id in a single step:
            - :meth:`._set_docs`
            - :meth:`._del_docs`
            - :meth:`._del_all_docs`


    """

    # Getitem APIs

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        return self._get_doc_by_id(self._offset2ids.get_id(offset))

    @abstractmethod
    def _get_doc_by_id(self, _id: str) -> 'Document':
        ...

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        """This function is derived from :meth:`_get_doc_by_offset`
        Override this function if there is a more efficient logic

        :param _slice: the slice used for indexing
        :return: an iterable of document
        """
        return self._get_docs_by_ids(self._offset2ids.get_id(_slice))

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

    def _del_doc_by_offset(self, offset: int):
        self._del_doc_by_id(self._offset2ids.get_id(offset))
        self._offset2ids.delete_by_offset(offset)

    def _del_doc(self, _id: str):
        self._offset2ids.delete_by_id(_id)
        self._del_doc_by_id(_id)

    @abstractmethod
    def _del_doc_by_id(self, _id: str):
        ...

    def _del_docs_by_slice(self, _slice: slice):
        """This function is derived and may not have the most efficient implementation.
        Override this function if there is a more efficient logic
        :param _slice: the slice used for indexing
        """
        ids = self._offset2ids.get_id(_slice)
        self._del_docs(ids)

    def _del_docs_by_mask(self, mask: Sequence[bool]):
        """This function is derived and may not have the most efficient implementation.
        Override this function if there is a more efficient logic
        :param mask: the boolean mask used for indexing
        """
        ids = list(itertools.compress(self._offset2ids, (_i for _i in mask)))
        self._del_docs(ids)

    def _del_all_docs(self):
        self._clear_storage()
        self._offset2ids = Offset2ID()

    def _del_docs_by_ids(self, ids):
        """This function is derived from :meth:`_del_doc_by_id`
        Override this function if there is a more efficient logic

        :param ids: the ids used for indexing
        """
        for _id in ids:
            self._del_doc_by_id(_id)

    def _del_docs(self, ids):
        self._del_docs_by_ids(ids)
        self._offset2ids.delete_by_ids(ids)

    def _clear_storage(self):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic.
        If you override this method, you should only take care of clearing the storage backend."""
        for doc in self:
            self._del_doc_by_id(doc.id)

    # Setitem API

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        self._set_doc(self._offset2ids.get_id(offset), value)

    def _set_doc(self, _id: str, value: 'Document'):
        if _id != value.id:
            self._offset2ids.update(self._offset2ids.index(_id), value.id)
        self._set_doc_by_id(_id, value)

    @abstractmethod
    def _set_doc_by_id(self, _id: str, value: 'Document'):
        ...

    def _set_docs_by_ids(self, ids, docs: Iterable['Document'], mismatch_ids: Dict):
        """This function is derived from :meth:`_set_doc_by_id`
        Override this function if there is a more efficient logic

        :param ids: the ids used for indexing
        """
        for _id, doc in zip(ids, docs):
            self._set_doc_by_id(_id, doc)

    def _set_docs(self, ids, docs: Iterable['Document']):
        docs = list(docs)
        mismatch_ids = {_id: doc.id for _id, doc in zip(ids, docs) if _id != doc.id}
        self._set_docs_by_ids(ids, docs, mismatch_ids)
        self._offset2ids.update_ids(mismatch_ids)

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

        ids = self._offset2ids.get_id(_slice)
        self._set_docs(ids, value)

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
            self._set_doc(_d.id, _v)

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
                self._set_doc(root_d.id, root_d)

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        :param offset: the offset used for indexing
        :param attr: the attribute of document to update
        :param value: the value doc's attr will be updated to
        """
        if attr == 'id' and value is None:
            raise ValueError(
                'setting the ID of a Document stored in a DocumentArray to None is not allowed'
            )
        _id = self._offset2ids.get_id(offset)
        d = self._get_doc_by_id(_id)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc(_id, d)

    def _set_doc_attr_by_id(self, _id: str, attr: str, value: Any):
        """This function is derived and may not have the most efficient implementation.

        Override this function if there is a more efficient logic
        :param _id: the id used for indexing
        :param attr: the attribute of document to update
        :param value: the value doc's attr will be updated to
        """
        if attr == 'id' and value is None:
            raise ValueError(
                'setting the ID of a Document stored in a DocumentArray to None is not allowed'
            )

        d = self._get_doc_by_id(_id)
        if hasattr(d, attr):
            setattr(d, attr, value)
            self._set_doc(_id, d)

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

    @abstractmethod
    def _load_offset2ids(self):
        ...

    @abstractmethod
    def _save_offset2ids(self):
        ...

    def __del__(self):
        if hasattr(self, '_offset2ids'):
            self._save_offset2ids()
