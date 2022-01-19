import base64
import io
import os.path
import pickle
from contextlib import nullcontext
from typing import Union, BinaryIO, TYPE_CHECKING, Type, Optional, Generator

from ....helper import random_uuid, __windows__, get_compress_ctx, decompress_bytes

if TYPE_CHECKING:
    from ....types import T
    from ....proto.docarray_pb2 import DocumentArrayProto
    from .... import Document, DocumentArray


class BinaryIOMixin:
    """Save/load an array to a binary file."""

    @classmethod
    def load_binary(
        cls: Type['T'],
        file: Union[str, BinaryIO, bytes],
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
        streaming: bool = False,
    ) -> Union['DocumentArray', Generator['Document', None, None]]:
        """Load array elements from a compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.
        In case protocol is pickle the `Documents` are streamed from disk to save memory usage
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
        if streaming:
            return cls._load_binary_stream(
                file_ctx,
                protocol=protocol,
                compress=compress,
                _show_progress=_show_progress,
            )
        else:
            return cls._load_binary_all(file_ctx, protocol, compress, _show_progress)

    @classmethod
    def _load_binary_stream(
        cls: Type['T'],
        file_ctx: str,
        protocol=None,
        compress=None,
        _show_progress=False,
    ) -> Generator['Document', None, None]:
        """Yield `Document` objects from a binary file

        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: a generator of `Document` objects
        """

        from .... import Document

        if _show_progress:
            from rich.progress import track as _track

            track = lambda x: _track(x, description='Deserializing')
        else:
            track = lambda x: x

        with file_ctx as f:
            version_numdocs_lendoc0 = f.read(9)
            # 1 byte (uint8)
            version = int.from_bytes(version_numdocs_lendoc0[0:1], 'big', signed=False)
            # 8 bytes (uint64)
            num_docs = int.from_bytes(version_numdocs_lendoc0[1:9], 'big', signed=False)

            for _ in track(range(num_docs)):
                # 4 bytes (uint32)
                len_current_doc_in_bytes = int.from_bytes(
                    f.read(4), 'big', signed=False
                )
                yield Document.from_bytes(
                    f.read(len_current_doc_in_bytes),
                    protocol=protocol,
                    compress=compress,
                )

    @classmethod
    def _load_binary_all(cls, file_ctx, protocol, compress, show_progress):
        """Read a `DocumentArray` object from a binary file

        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: a `DocumentArray`
        """
        from .... import Document

        with file_ctx as fp:
            d = fp.read() if hasattr(fp, 'read') else fp

        if protocol == 'pickle-array' or protocol == 'protobuf-array':
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

        # Binary format for streaming case
        else:
            # 1 byte (uint8)
            version = int.from_bytes(d[0:1], 'big', signed=False)
            # 8 bytes (uint64)
            num_docs = int.from_bytes(d[1:9], 'big', signed=False)
            if show_progress:
                from rich.progress import track as _track

                track = lambda x: _track(x, description='Deserializing')
            else:
                track = lambda x: x

            # this 9 is version + num_docs bytes used
            start_pos = 9
            docs = []

            for _ in track(range(num_docs)):
                # 4 bytes (uint32)
                len_current_doc_in_bytes = int.from_bytes(
                    d[start_pos : start_pos + 4], 'big', signed=False
                )
                start_doc_pos = start_pos + 4
                end_doc_pos = start_doc_pos + len_current_doc_in_bytes
                start_pos = end_doc_pos

                # variable length bytes doc
                doc = Document.from_bytes(
                    d[start_doc_pos:end_doc_pos], protocol=protocol, compress=compress
                )
                docs.append(doc)

            return cls(docs)

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

        if protocol == 'protobuf-array' or protocol == 'pickle-array':
            compress_ctx = get_compress_ctx(compress, mode='wb')
        else:
            compress_ctx = None

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
                elif protocol in ('pickle', 'protobuf'):
                    # Binary format for streaming case
                    if _show_progress:
                        from rich.progress import track as _track

                        track = lambda x: _track(x, description='Serializing')
                    else:
                        track = lambda x: x

                    # V1 DocArray streaming serialization format
                    # | 1 byte | 8 bytes | 4 bytes | variable | 4 bytes | variable ...

                    # 1 byte (uint8)
                    version_byte = b'\x01'
                    # 8 bytes (uint64)
                    num_docs_as_bytes = len(self).to_bytes(8, 'big', signed=False)
                    f.write(version_byte + num_docs_as_bytes)

                    for d in track(self):
                        # 4 bytes (uint32)
                        doc_as_bytes = d.to_bytes(protocol=protocol, compress=compress)

                        # variable size bytes
                        len_doc_as_bytes = len(doc_as_bytes).to_bytes(
                            4, 'big', signed=False
                        )
                        f.write(len_doc_as_bytes + doc_as_bytes)
                else:
                    raise ValueError(
                        f'protocol={protocol} is not supported. Can be only `protobuf`,`pickle`,`protobuf-array`,`pickle-array`.'
                    )

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
