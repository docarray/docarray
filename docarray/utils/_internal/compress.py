from typing import IO, TYPE_CHECKING, Callable, Optional

from docarray.utils._internal.misc import import_library


def _compress_bytes(data: bytes, algorithm: Optional[str] = None) -> bytes:
    if algorithm == 'lz4':
        if TYPE_CHECKING:
            from lz4 import frame
        else:
            lz4 = import_library('lz4', raise_error=True)  # noqa: F841
            from lz4 import frame

        data = frame.compress(data)
    elif algorithm == 'bz2':
        import bz2

        data = bz2.compress(data)
    elif algorithm == 'lzma':
        import lzma

        data = lzma.compress(data)
    elif algorithm == 'zlib':
        import zlib

        data = zlib.compress(data)
    elif algorithm == 'gzip':
        import gzip

        data = gzip.compress(data)
    return data


def _decompress_bytes(data: bytes, algorithm: Optional[str] = None) -> bytes:
    if algorithm == 'lz4':
        if TYPE_CHECKING:
            from lz4 import frame
        else:
            lz4 = import_library('lz4', raise_error=True)  # noqa: F841
            from lz4 import frame

        data = frame.decompress(data)
    elif algorithm == 'bz2':
        import bz2

        data = bz2.decompress(data)
    elif algorithm == 'lzma':
        import lzma

        data = lzma.decompress(data)
    elif algorithm == 'zlib':
        import zlib

        data = zlib.decompress(data)
    elif algorithm == 'gzip':
        import gzip

        data = gzip.decompress(data)
    return data


def _get_compress_ctx(algorithm: Optional[str] = None) -> Optional[Callable]:
    if algorithm == 'lz4':
        if TYPE_CHECKING:
            from lz4 import frame
        else:
            lz4 = import_library('lz4', raise_error=True)  # noqa: F841
            from lz4 import frame

        def _fun(x: IO[bytes]):
            return frame.LZ4FrameFile(x, 'wb')

        compress_ctx = _fun
    elif algorithm == 'gzip':
        import gzip

        def _fun(x: IO[bytes]):
            return gzip.GzipFile(fileobj=x, mode='wb')

        compress_ctx = _fun
    elif algorithm == 'bz2':
        import bz2

        def _fun(x: IO[bytes]):
            return bz2.BZ2File(filename=x, mode='wb')

        compress_ctx = _fun
    elif algorithm == 'lzma':
        import lzma

        def _fun(x: IO[bytes]):
            return lzma.LZMAFile(filename=x, mode='wb')

        compress_ctx = _fun
    else:
        compress_ctx = None
    return compress_ctx
