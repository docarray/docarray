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
            raise ValueError(
                f'Failed to initialize Document from obj={obj}, kwargs={kwargs}'
            )
