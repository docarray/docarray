from typing import TYPE_CHECKING, Optional

from .data import DocumentData, default_values
from .mixins import AllMixins

if TYPE_CHECKING:
    from .score import NamedScore


class Document(AllMixins):
    _all_doc_content_keys = {'content', 'blob', 'text', 'buffer'}

    def __init__(self, obj: Optional['Document'] = None, copy: bool = False, **kwargs):
        self._data = None
        if isinstance(obj, Document):
            if copy:
                self.copy_from(obj)
            else:
                self._data = obj._data
        elif isinstance(obj, dict):
            kwargs.update(obj)
        if kwargs:
            # check if there are mutually exclusive content fields
            if len(self._all_doc_content_keys.intersection(kwargs.keys())) > 1:
                raise ValueError(
                    f'Document content fields are mutually exclusive, please provide only one of {self._all_doc_content_keys}'
                )
            self._data = DocumentData(self, **kwargs)
        if obj is None and not kwargs:
            self._data = DocumentData(self)

        if self._data is None:
            raise ValueError(f'Failed to initialize Document from obj={obj}, kwargs={kwargs}')

    def _set_default_value_if_none(self, key):
        if getattr(self._data, key) is None:
            v = default_values.get(key, None)
            if v is not None:
                if v == 'DocumentArray':
                    from .. import DocumentArray
                    setattr(self._data, key, DocumentArray())
                elif v == 'ChunkArray':
                    from ..array.chunk import ChunkArray
                    setattr(self._data, key, ChunkArray(None, reference_doc=self))
                elif v == 'MatchArray':
                    from ..array.match import MatchArray
                    setattr(self._data, key, MatchArray(None, reference_doc=self))
                else:
                    setattr(self._data, key, v() if callable(v) else v)

    def __setattr__(self, key, value):
        if value is not None:
            if key == 'text' or key == 'blob' or key == 'buffer':
                # enable mutual exclusivity for content field
                if value != default_values.get(key):
                    self.text = None
                    self.blob = None
                    self.buffer = None
            elif key == 'chunks':
                from ..array.chunk import ChunkArray
                if not isinstance(value, ChunkArray):
                    value = ChunkArray(None, reference_doc=self)
            elif key == 'matches':
                from ..array.match import MatchArray
                if not isinstance(value, MatchArray):
                    value = MatchArray(None, reference_doc=self)
        super().__setattr__(key, value)