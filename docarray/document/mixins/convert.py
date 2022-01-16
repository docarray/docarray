from typing import Optional, TYPE_CHECKING

import numpy as np

from .helper import _uri_to_blob, _to_datauri, _is_datauri

if TYPE_CHECKING:
    from ...types import T


class ConvertMixin:
    """Provide helper functions for :class:`Document` to support conversion between :attr:`.tensor`, :attr:`.text`
    and :attr:`.blob`."""

    def convert_blob_to_tensor(
        self: 'T', dtype: Optional[str] = None, count: int = -1, offset: int = 0
    ) -> 'T':
        """Assuming the :attr:`blob` is a _valid_ buffer of Numpy ndarray,
        set :attr:`tensor` accordingly.

        :param dtype: Data-type of the returned array; default: float.
        :param count: Number of items to read. ``-1`` means all data in the buffer.
        :param offset: Start reading the buffer from this offset (in bytes); default: 0.

        :return: itself after processed
        """
        self.tensor = np.frombuffer(self.blob, dtype=dtype, count=count, offset=offset)
        return self

    def convert_tensor_to_blob(self: 'T') -> 'T':
        """Convert :attr:`.tensor` to :attr:`.blob` inplace.

        :return: itself after processed
        """
        self.blob = self.tensor.tobytes()
        return self

    def convert_uri_to_datauri(
        self: 'T', charset: str = 'utf-8', base64: bool = False
    ) -> 'T':
        """Convert :attr:`.uri` to dataURI and store it in :attr:`.uri` inplace.

        :param charset: charset may be any character set registered with IANA
        :param base64: used to encode arbitrary octet sequences into a form that satisfies the rules of 7bit. Designed to be efficient for non-text 8 bit and binary data. Sometimes used for text data that frequently uses non-US-ASCII characters.

        :return: itself after processed
        """
        if not _is_datauri(self.uri):
            blob = _uri_to_blob(self.uri)
            self.uri = _to_datauri(self.mime_type, blob, charset, base64, binary=True)
        return self
