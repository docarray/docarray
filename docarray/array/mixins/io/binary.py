import base64
import io
import os
import os.path
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Union, BinaryIO, TYPE_CHECKING, Type, Optional, Generator

from ....helper import (
    get_compress_ctx,
    decompress_bytes,
    protocol_and_compress_from_file_path,
)

if TYPE_CHECKING:
    from ....typing import T
    from ....proto.docarray_pb2 import DocumentArrayProto
    from .... import Document, DocumentArray


class LazyRequestReader:
    def __init__(self, r):
        self._data = r.iter_content(chunk_size=1024 * 1024)
        self.content = b''

    def __getitem__(self, item: slice):
        while len(self.content) < item.stop:
            try:
                self.content += next(self._data)
            except StopIteration:
                return self.content[item.start : -1 : item.step]
        return self.content[item]


class BinaryIOMixin:
    """Save/load an array to a binary file."""

    @classmethod
    def load_binary(
        cls: Type['T'],
        file: Union[str, BinaryIO, bytes, Path],
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
        streaming: bool = False,
        *args,
        **kwargs,
    ) -> Union['DocumentArray', Generator['Document', None, None]]:
        """Load array elements from a compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.
        In case protocol is pickle the `Documents` are streamed from disk to save memory usage
        :return: a DocumentArray object

        .. note::
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be loaded assuming `protocol=protobuf`
            and `compress=lz4`.
        """

        if isinstance(file, (io.BufferedReader, LazyRequestReader)):
            file_ctx = nullcontext(file)
        elif isinstance(file, bytes):
            file_ctx = nullcontext(file)
        # by checking path existence we allow file to be of type Path, LocalPath, PurePath and str
        elif os.path.exists(file):
            protocol, compress = protocol_and_compress_from_file_path(
                file, protocol, compress
            )
            file_ctx = open(file, 'rb')
        else:
            raise FileNotFoundError(f'cannot find file {file}')
        if streaming:
            return cls._load_binary_stream(
                file_ctx,
                protocol=protocol,
                compress=compress,
                _show_progress=_show_progress,
            )
        else:
            return cls._load_binary_all(
                file_ctx, protocol, compress, _show_progress, *args, **kwargs
            )

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

        from .pbar import get_progressbar
        from rich import filesize

        with file_ctx as f:
            version_numdocs_lendoc0 = f.read(9)
            # 1 byte (uint8)
            version = int.from_bytes(version_numdocs_lendoc0[0:1], 'big', signed=False)
            # 8 bytes (uint64)
            num_docs = int.from_bytes(version_numdocs_lendoc0[1:9], 'big', signed=False)

            pbar, t = get_progressbar(
                'Deserializing', disable=not _show_progress, total=num_docs
            )

            with pbar:
                _total_size = 0
                pbar.start_task(t)
                for _ in range(num_docs):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        f.read(4), 'big', signed=False
                    )
                    _total_size += len_current_doc_in_bytes
                    yield Document.from_bytes(
                        f.read(len_current_doc_in_bytes),
                        protocol=protocol,
                        compress=compress,
                    )
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )

    @classmethod
    def _load_binary_all(
        cls, file_ctx, protocol, compress, show_progress, *args, **kwargs
    ):
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
            from rich import filesize
            from .pbar import get_progressbar

            # 1 byte (uint8)
            version = int.from_bytes(d[0:1], 'big', signed=False)
            # 8 bytes (uint64)
            num_docs = int.from_bytes(d[1:9], 'big', signed=False)

            pbar, t = get_progressbar(
                'Deserializing', disable=not show_progress, total=num_docs
            )

            # this 9 is version + num_docs bytes used
            start_pos = 9
            docs = []
            with pbar:
                _total_size = 0
                pbar.start_task(t)

                for _ in range(num_docs):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        d[start_pos : start_pos + 4], 'big', signed=False
                    )
                    start_doc_pos = start_pos + 4
                    end_doc_pos = start_doc_pos + len_current_doc_in_bytes
                    start_pos = end_doc_pos

                    # variable length bytes doc
                    doc = Document.from_bytes(
                        d[start_doc_pos:end_doc_pos],
                        protocol=protocol,
                        compress=compress,
                    )
                    docs.append(doc)
                    _total_size += len_current_doc_in_bytes
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )
            return cls(docs, *args, **kwargs)

    @classmethod
    def from_bytes(
        cls: Type['T'],
        data: bytes,
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
        *args,
        **kwargs,
    ) -> 'T':
        return cls.load_binary(
            data,
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
            *args,
            **kwargs,
        )

    def save_binary(
        self,
        file: Union[str, BinaryIO],
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
    ) -> None:
        """Save array elements into a binary file.

        :param file: File or filename to which the data is saved.
        :param protocol: protocol to use
        :param compress: compress algorithm to use

         .. note::
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be created using `protocol=protobuf`
            and `compress=lz4`.

        Comparing to :meth:`save_json`, it is faster and the file is smaller, but not human-readable.

        .. note::
            To get a binary presentation in memory, use ``bytes(...)``.

        """
        if isinstance(file, io.BufferedWriter):
            file_ctx = nullcontext(file)
        else:
            _protocol, _compress = protocol_and_compress_from_file_path(file)

            if _protocol is not None:
                protocol = _protocol
            if _compress is not None:
                compress = _compress

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
            # delegate the compression to per-doc compression
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
                    from rich import filesize
                    from .pbar import get_progressbar

                    pbar, t = get_progressbar(
                        'Serializing', disable=not _show_progress, total=len(self)
                    )

                    f.write(self._stream_header)

                    with pbar:
                        _total_size = 0
                        pbar.start_task(t)
                        for d in self:
                            r = d._to_stream_bytes(protocol=protocol, compress=compress)
                            f.write(r)
                            _total_size += len(r)
                            pbar.update(
                                t,
                                advance=1,
                                total_size=str(filesize.decimal(_total_size)),
                            )
                else:
                    raise ValueError(
                        f'protocol={protocol} is not supported. Can be only `protobuf`, `pickle`, `protobuf-array`, `pickle-array`.'
                    )

            if not _file_ctx:
                return bf.getvalue()

    def to_protobuf(self, ndarray_type: Optional[str] = None) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message.

        :param ndarray_type: can be ``list`` or ``numpy``, if set it will force all ndarray-like object from all
            Documents to ``List`` or ``numpy.ndarray``.
        :return: the protobuf message
        """
        from ....proto.docarray_pb2 import DocumentArrayProto

        dap = DocumentArrayProto()
        for d in self:
            dap.docs.append(d.to_protobuf(ndarray_type))
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
        *args,
        **kwargs,
    ) -> 'T':
        return cls.load_binary(
            base64.b64decode(data),
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
            *args,
            **kwargs,
        )

    def to_base64(
        self,
        protocol: str = 'pickle-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> str:
        return base64.b64encode(self.to_bytes(protocol, compress)).decode('utf-8')

    @property
    def _stream_header(self) -> bytes:
        # Binary format for streaming case

        # V1 DocArray streaming serialization format
        # | 1 byte | 8 bytes | 4 bytes | variable | 4 bytes | variable ...

        # 1 byte (uint8)
        version_byte = b'\x01'
        # 8 bytes (uint64)
        num_docs_as_bytes = len(self).to_bytes(8, 'big', signed=False)
        return version_byte + num_docs_as_bytes
