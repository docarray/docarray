from typing import Union, BinaryIO, TYPE_CHECKING, Type, Optional, Generator
from docarray.array.mixins import BinaryIOMixin

if TYPE_CHECKING:
    from ....types import T
    from .... import Document, DocumentArray


def _check_protocol(protocol):
    if protocol == 'pickle-array':
        raise ValueError(
            'protocol pickle-array is not supported for DocumentArraySqlite'
        )


class SqliteBinaryIOMixin(BinaryIOMixin):
    """Save/load an array to a binary file."""

    @classmethod
    def load_binary(
        cls: Type['T'],
        file: Union[str, BinaryIO, bytes],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
        streaming: bool = False,
    ) -> Union['DocumentArray', Generator['Document', None, None]]:
        """Load array elements from a compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use. 'pickle-array' is not supported for DocumentArraySqlite
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.
        In case protocol is pickle the `Documents` are streamed from disk to save memory usage
        :return: a DocumentArray object
        """
        _check_protocol(protocol)
        return super().load_binary(
            file=file,
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
            streaming=streaming,
        )

    @classmethod
    def from_bytes(
        cls: Type['T'],
        data: bytes,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> 'T':
        _check_protocol(protocol)
        return super().from_bytes(
            data=data,
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
        )

    def save_binary(
        self,
        file: Union[str, BinaryIO],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
    ) -> None:
        """Save array elements into a binary file.

        Comparing to :meth:`save_json`, it is faster and the file is smaller, but not human-readable.

        .. note::
            To get a binary presentation in memory, use ``bytes(...)``.

        :param protocol: protocol to use. 'pickle-array' is not supported for DocumentArraySqlite
        :param compress: compress algorithm to use
        :param file: File or filename to which the data is saved.
        """
        _check_protocol(protocol)
        super(SqliteBinaryIOMixin, self).save_binary(
            file=file, protocol=protocol, compress=compress
        )

    def to_bytes(
        self,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        _file_ctx: Optional[BinaryIO] = None,
        _show_progress: bool = False,
    ) -> bytes:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param _file_ctx: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use. 'pickle-array' is not supported for DocumentArraySqlite
        :param compress: compress algorithm to use
        :param _show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the binary serialization in bytes
        """
        _check_protocol(protocol)
        return super(SqliteBinaryIOMixin, self).to_bytes(
            protocol=protocol,
            compress=compress,
            _file_ctx=_file_ctx,
            _show_progress=_show_progress,
        )

    @classmethod
    def from_base64(
        cls: Type['T'],
        data: str,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> 'T':
        _check_protocol(protocol)
        return super().from_base64(
            data=data,
            protocol=protocol,
            compress=compress,
            _show_progress=_show_progress,
        )

    def to_base64(
        self,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        _show_progress: bool = False,
    ) -> str:
        return super(SqliteBinaryIOMixin, self).to_base64(
            protocol=protocol, compress=compress, _show_progress=_show_progress
        )
