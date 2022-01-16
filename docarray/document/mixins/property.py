import mimetypes
from typing import TYPE_CHECKING, Optional

from ._property import _PropertyMixin

if TYPE_CHECKING:
    from ...types import DocumentContentType

_all_mime_types = set(mimetypes.types_map.values())


class PropertyMixin(_PropertyMixin):
    def _clear_content(self):
        self._data.content = None
        self._data.text = None
        self._data.tensor = None
        self._data.blob = None

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
            self.blob = value
        elif isinstance(value, str):
            self.text = value
        else:
            self.tensor = value

    @property
    def content_type(self) -> Optional[str]:
        nf = self.non_empty_fields
        if 'text' in nf:
            return 'text'
        elif 'tensor' in nf:
            return 'tensor'
        elif 'blob' in nf:
            return 'blob'
