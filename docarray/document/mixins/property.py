from typing import TYPE_CHECKING, Optional

from ._property import _PropertyMixin

if TYPE_CHECKING:
    from ...types import ArrayType, DocumentContentType


class PropertyMixin(_PropertyMixin):

    def _clear_content(self):
        self._data.text = None
        self._data.blob = None
        self._data.buffer = None

    @_PropertyMixin.text.setter
    def text(self, value: str):
        self._clear_content()
        self._data.text = value

    @_PropertyMixin.blob.setter
    def blob(self, value: 'ArrayType'):
        self._clear_content()
        self._data.blob = value

    @_PropertyMixin.buffer.setter
    def buffer(self, value: bytes):
        self._clear_content()
        self._data.buffer = value

    @property
    def content(self) -> Optional['DocumentContentType']:
        ct = self.content_type
        if ct:
            return getattr(self, ct)

    @content.setter
    def content(self, value: 'DocumentContentType'):
        if value is None:
            self._clear_content()
        elif isinstance(value, bytes):
            self.buffer = value
        elif isinstance(value, str):
            self.text = value
        else:
            self.blob = value

    @property
    def content_type(self) -> Optional[str]:
        nf = self.non_empty_fields
        if 'text' in nf:
            return 'text'
        elif 'blob' in nf:
            return 'blob'
        elif 'buffer' in nf:
            return 'buffer'