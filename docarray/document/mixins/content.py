from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...typing import T

_DIGEST_SIZE = 8


class ContentPropertyMixin:
    """Provide helper functions for :class:`Document` to allow universal content property access. """

    @property
    def content_type(self) -> Optional[str]:
        if self._pb_body.text is not None:
            return 'text'
        if self._pb_body.buffer is not None:
            return 'buffer'
        if self._pb_body.blob is not None:
            return 'blob'

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

    def _clear_content(self):
        self._doc_data.text = None
        self._doc_data.blob = None
        self._doc_data.buffer = None
