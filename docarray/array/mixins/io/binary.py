import io
import os.path
from contextlib import nullcontext
from typing import Union, BinaryIO, TYPE_CHECKING, Type, Optional

from ....helper import random_uuid, __windows__, get_compress_ctx, decompress_bytes

if TYPE_CHECKING:
    from ....types import T


class BinaryIOMixin:
    """Save/load an array to a binary file. """

    @classmethod
    def load_binary(
        cls: Type['T'],
        file: Union[str, BinaryIO, bytes],
        protocol: Union[str, int] = 'protobuf',
        compress: Optional[str] = None,
    ) -> 'T':
        """Load array elements from a LZ4-compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.

        :return: a DocumentArray object
        """

        if isinstance(file, io.BufferedReader):
            file_ctx = nullcontext(file)
        elif isinstance(file, bytes):
            file_ctx = nullcontext(file)
        elif os.path.exists(file):
            file_ctx = open(file, 'rb')
        else:
            raise ValueError(f'unsupported input {file!r}')

        from .... import Document

        with file_ctx as fp:
            d = fp.read() if hasattr(fp, 'read') else fp
            if get_compress_ctx(algorithm=compress) is not None:
                d = decompress_bytes(d, algorithm=compress)
                compress = None

            _len = len(random_uuid().bytes)
            _binary_delimiter = d[:_len]  # first get delimiter
            da = cls()
            da.extend(
                Document.from_bytes(od, protocol=protocol, compress=compress)
                for od in d[_len:].split(_binary_delimiter)
            )
            return da

    @classmethod
    def from_bytes(
        cls: Type['T'],
        data: bytes,
        protocol: Union[str, int] = 'protobuf',
        compress: Optional[str] = None,
    ) -> 'T':
        return cls.load_binary(data, protocol=protocol, compress=compress)

    def save_binary(
        self,
        file: Union[str, BinaryIO],
        protocol: Union[str, int] = 'protobuf',
        compress: Optional[str] = None,
    ) -> None:
        """Save array elements into a LZ4 compressed binary file.

        Comparing to :meth:`save_json`, it is faster and the file is smaller, but not human-readable.

        .. note::
            To get a binary presentation in memory, use ``bytes(...)``.

        :param file: File or filename to which the data is saved.
        """
        if isinstance(file, io.BufferedWriter):
            file_ctx = nullcontext(file)
        else:
            if __windows__:
                file_ctx = open(file, 'wb', newline='')
            else:
                file_ctx = open(file, 'wb')

        self.to_bytes(protocol=protocol, compress=compress, _file_ctx=file_ctx)

    def to_bytes(
        self,
        protocol: Union[str, int] = 'protobuf',
        compress: Optional[str] = None,
        _file_ctx: Optional[BinaryIO] = None,
    ) -> bytes:
        """Serialize itself into bytes with LZ4 compression.

        For more Pythonic code, please use ``bytes(...)``.

        :return: the binary serialization in bytes
        """

        _binary_delimiter = random_uuid().bytes
        compress_ctx = get_compress_ctx(compress, mode='wb')
        with (_file_ctx or io.BytesIO()) as bf:
            if compress_ctx is None:
                # if compress do not support streaming then postpone the compress
                # into the for-loop
                f, fc = bf, nullcontext()
            else:
                f = compress_ctx(bf)
                fc = f
                compress = None
            with fc:
                for d in self:
                    f.write(_binary_delimiter)
                    f.write(d.to_bytes(protocol=protocol, compress=compress))
            if not _file_ctx:
                return bf.getvalue()

    def __bytes__(self):
        return self.to_bytes()
