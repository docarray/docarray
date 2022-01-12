import io
import os.path
import base64
import pickle
from contextlib import nullcontext
from typing import Union, BinaryIO, TYPE_CHECKING, Type, Optional

from ....helper import random_uuid, __windows__, get_compress_ctx, decompress_bytes

if TYPE_CHECKING:
    from ....types import T
    from ....proto.docarray_pb2 import DocumentArrayProto


class BinaryIOMixin:
    """Save/load an array to a binary file."""

    @classmethod
    def load_binary(
        cls: Type['T'],
        file: Union[str, BinaryIO, bytes],
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> 'T':
        """Load array elements from a LZ4-compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`

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

            if protocol == 'protobuf-array':
                from ....proto.docarray_pb2 import DocumentArrayProto

                dap = DocumentArrayProto()
                dap.ParseFromString(d)

                return cls.from_protobuf(dap)
            elif protocol == 'pickle-array':
                return pickle.loads(d)
            else:
                _len = len(random_uuid().bytes)
                _binary_delimiter = d[:_len]  # first get delimiter
                if _show_progress:
                    from rich.progress import track as _track

                    track = lambda x: _track(x, description='Deserializing')
                else:
                    track = lambda x: x
                return cls(
                    Document.from_bytes(od, protocol=protocol, compress=compress)
                    for od in track(d[_len:].split(_binary_delimiter))
                )

    @classmethod
    def from_bytes(
        cls: Type['T'],
        data: bytes,
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> 'T':
        return cls.load_binary(
            data, protocol=protocol, compress=compress, _show_progress=_show_progress
        )

    def save_binary(
        self,
        file: Union[str, BinaryIO],
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
    ) -> None:
        """Save array elements into a binary file.

        Comparing to :meth:`save_json`, it is faster and the file is smaller, but not human-readable.

        .. note::
            To get a binary presentation in memory, use ``bytes(...)``.

        :param protocol: protocol to use
        :param compress: compress algorithm to use
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
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _file_ctx: Optional[BinaryIO] = None,
        _show_progress: bool = False,
    ) -> bytes:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param _file_ctx: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
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
                if protocol == 'protobuf-array':
                    f.write(self.to_protobuf().SerializePartialToString())
                elif protocol == 'pickle-array':
                    f.write(pickle.dumps(self))
                else:
                    if _show_progress:
                        from rich.progress import track as _track

                        track = lambda x: _track(x, description='Serializing')
                    else:
                        track = lambda x: x

                    for d in track(self):
                        f.write(_binary_delimiter)
                        f.write(d.to_bytes(protocol=protocol, compress=compress))
            if not _file_ctx:
                return bf.getvalue()

    def to_protobuf(self) -> 'DocumentArrayProto':
        from ....proto.docarray_pb2 import DocumentArrayProto

        dap = DocumentArrayProto()
        for d in self:
            dap.docs.append(d.to_protobuf())
        return dap

    @classmethod
    def from_protobuf(cls: Type['T'], pb_msg: 'DocumentArrayProto') -> 'T':
        from .... import Document

        return cls(Document.from_protobuf(od) for od in pb_msg.docs)

    def __bytes__(self):
        return self.to_bytes()

    @classmethod
    def from_base64(
        cls: Type['T'],
        data: str,
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> 'T':
        return cls.load_binary(
            base64.b64decode(data),
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
        )

    def to_base64(
        self,
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> str:
        return base64.b64encode(self.to_bytes(protocol, compress)).decode('utf-8')
