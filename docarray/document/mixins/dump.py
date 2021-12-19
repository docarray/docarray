from typing import Union, BinaryIO, TYPE_CHECKING

from .helper import _uri_to_buffer, _get_file_context

if TYPE_CHECKING:
    from ...types import T


class UriFileMixin:
    """Provide helper functions for :class:`Document` to dump content to a file. """

    def save_uri_to_file(self: 'T', file: Union[str, BinaryIO]) -> 'T':
        """Save :attr:`.uri` into a file

        :param file: File or filename to which the data is saved.

        :return: itself after processed
        """
        fp = _get_file_context(file)
        with fp:
            buffer = _uri_to_buffer(self.uri)
            fp.write(buffer)
        return self
