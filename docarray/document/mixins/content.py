from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import T


class ContentPropertyMixin:
    """Provide helper functions for :class:`Document` to allow universal content property access. """

    @property
    def content_hash(self) -> int:
        """Get the document hash according to its content.

        :return: the unique hash code to represent this Document
        """
        return hash(self)

    def convert_content_to_datauri(self: 'T') -> 'T':
        """Convert :attr:`.content` in :attr:`.uri` inplace with best effort

        :return: itself after processed
        """
        if self.text:
            self.convert_text_to_datauri()
        elif self.blob:
            self.convert_blob_to_datauri()
        elif self.content_type:
            raise NotImplementedError
        return self
