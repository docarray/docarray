import mimetypes
from typing import TYPE_CHECKING, Optional

from ._property import _PropertyMixin

if TYPE_CHECKING:
    from ...types import DocumentContentType

_all_mime_types = set(mimetypes.types_map.values())


class PropertyMixin(_PropertyMixin):

    def _clear_content(self):
        self._data.text = None
        self._data.blob = None
        self._data.buffer = None

    @_PropertyMixin.mime_type.setter
    def mime_type(self, value: str):
        if value in _all_mime_types:
            self._data.mime_type = value
        elif value:
            r = mimetypes.guess_type(f'*.{value}')[0]
            self._data.mime_type = r or value

    @_PropertyMixin.uri.setter
    def uri(self, value: str):
        mime_type = mimetypes.guess_type(value)[0]
        if mime_type:
            self._data.mime_type = mime_type

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
