from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...typing import T, DocumentContentType

_DIGEST_SIZE = 8


class ContentPropertyMixin:
    """Provide helper functions for :class:`Document` to allow universal content property access. """

    @property
    def content(self) -> Optional['DocumentContentType']:
        ct = self.content_type
        if ct:
            return getattr(self, ct)

    @content.setter
    def content(self, value: Optional['DocumentContentType']):
        if value is None:
            self.text = None
            self.blob = None
            self.buffer = None
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

    @property
    def content_hash(self) -> int:
        """Get the document hash according to its content.

        :return: the unique hash code to represent this Document
        """
        return hash(self)

    def dump_content_to_datauri(self: 'T') -> 'T':
        """Convert :attr:`.content` in :attr:`.uri` inplace with best effort

        :return: itself after processed
        """
        if self.text:
            self.convert_text_to_uri()
        elif self.buffer:
            self.convert_buffer_to_uri()
        elif self.content_type:
            raise NotImplementedError
        return self
