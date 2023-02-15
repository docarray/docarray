from typing import Optional, Callable, Any


def _compress_bytes(data: bytes, algorithm: Optional[str] = None) -> bytes:
    if algorithm == 'lz4':
        import lz4.frame  # type: ignore

        data = lz4.frame.compress(data)
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
        import lz4.frame  # type: ignore

        data = lz4.frame.decompress(data)
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


def _get_compress_ctx(
    algorithm: Optional[str] = None, mode: str = 'wb'
) -> Optional[Callable]:
    if algorithm == 'lz4':
        import lz4.frame  # type: ignore

        def _fun(x: str):
            return lz4.frame.LZ4FrameFile(x, mode)

        compress_ctx = _fun
    elif algorithm == 'gzip':
        import gzip

        def _fun(x: str):
            return gzip.GzipFile(fileobj=x, mode=mode)

        compress_ctx = _fun
    elif algorithm == 'bz2':
        import bz2

        def _fun(x: str):
            return bz2.BZ2File(filename=x, mode=mode)

        compress_ctx = _fun
    elif algorithm == 'lzma':
        import lzma

        def _fun(x: str):
            return lzma.LZMAFile(filename=x, mode=mode)

        compress_ctx = _fun
    else:
        compress_ctx = None
    return compress_ctx
