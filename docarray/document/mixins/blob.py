from typing import TYPE_CHECKING, Union, BinaryIO

from .helper import _uri_to_blob, _to_datauri, _get_file_context

if TYPE_CHECKING:
    from ...types import T


class BlobDataMixin:
    """Provide helper functions for :class:`Document` to handle binary data. """

    def load_uri_to_blob(self: 'T') -> 'T':
        """Convert :attr:`.uri` to :attr:`.blob` inplace.
        Internally it downloads from the URI and set :attr:`blob`.

        :return: itself after processed
        """
        self.blob = _uri_to_blob(self.uri)
        return self

    def convert_blob_to_datauri(
        self: 'T', charset: str = 'utf-8', base64: bool = False
    ) -> 'T':
        """Convert :attr:`.blob` to data :attr:`.uri` in place.
        Internally it first reads into blob and then converts it to data URI.

        :param charset: charset may be any character set registered with IANA
        :param base64: used to encode arbitrary octet sequences into a form that satisfies the rules of 7bit.
            Designed to be efficient for non-text 8 bit and binary data. Sometimes used for text data that
            frequently uses non-US-ASCII characters.

        :return: itself after processed
        """

        if not self.mime_type:
            raise ValueError(
                f'{self.mime_type} is unset, can not convert it to data uri'
            )

        self.uri = _to_datauri(self.mime_type, self.blob, charset, base64, binary=True)
        return self

    def save_blob_to_file(self: 'T', file: Union[str, BinaryIO]) -> 'T':
        """Save :attr:`.blob` into a file

        :param file: File or filename to which the data is saved.
        :return: itself after processed
        """
        fp = _get_file_context(file)
        with fp:
            fp.write(self.blob)
        return self
