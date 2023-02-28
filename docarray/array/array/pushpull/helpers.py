# It is usually a bad idea to have a helper file because it means we don't know where to put the code (or haven't put much thought into it).
# With that said, rules are meant to be broken, we will live with this for now.
from typing import Dict, Iterable, Iterator, NoReturn, Optional, Sequence

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import requests

    from docarray import BaseDocument


def get_version_info() -> Dict:
    """
    Get the version of libraries used in Jina and environment variables.

    :return: Version information and environment variables
    """
    import platform
    from uuid import getnode

    import google.protobuf
    from google.protobuf.internal import api_implementation

    from docarray import __version__

    return {
        'docarray': __version__,
        'protobuf': google.protobuf.__version__,
        'proto-backend': api_implementation.Type(),
        'python': platform.python_version(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'platform-version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'uid': getnode(),
    }


def ibatch(iterable: Sequence, batch_size: int = 32) -> Iterable:
    """Get an iterator of batched items from Sequence."""
    seq_len = len(iterable)
    for offset in range(0, seq_len, batch_size):
        yield iterable[offset : min(offset + batch_size, seq_len)]


class _BufferedCachingReader:
    """A buffered reader that writes to a cache file while reading."""

    def __init__(
        self, iter_bytes: Iterator[bytes], cache_path: Optional['Path'] = None
    ):
        self._data = iter_bytes
        self._chunk: bytes = b''
        self._seek = 0
        self._chunk_len = 0

        self._cache = open(cache_path, 'wb') if cache_path else None

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return b''.join(self._data)

        if self._seek + size > self._chunk_len:
            _bytes = self._chunk[self._seek : self._chunk_len]
            size -= self._chunk_len - self._seek

            self._chunk = next(self._data)
            self._seek = 0
            self._chunk_len = len(self._chunk)
            if self._cache:
                self._cache.write(self._chunk)

            _bytes += self._chunk[self._seek : self._seek + size]
            self._seek += size
            return _bytes
        else:
            _bytes = self._chunk[self._seek : self._seek + size]
            self._seek += size
            return _bytes

    def __del__(self):
        if self._cache:
            self._cache.close()


class _BufferedCachingRequestReader(_BufferedCachingReader):
    """A buffered reader for requests.Response that writes to a cache file while reading."""

    def __init__(self, r: 'requests.Response', cache_path: Optional['Path'] = None):
        super().__init__(r.iter_content(chunk_size=2**20), cache_path)


def raise_req_error(resp: 'requests.Response') -> NoReturn:
    """Definitely raise an error from a response."""
    resp.raise_for_status()
    raise ValueError(f'Unexpected response status: {resp.status_code}')


def docs_to_binary_stream(
    docs: Iterator['BaseDocument'],
    protocol: str = 'protobuf',
    compress: Optional[str] = None,
    show_progress: bool = False,
    total: Optional[int] = None,
) -> Iterator[bytes]:
    from rich import filesize

    from docarray.utils.progress_bar import _get_progressbar

    if total is not None:
        pbar, t = _get_progressbar(
            'Serializing', disable=not show_progress, total=total
        )

    # Stream header
    if total:
        yield b'x01' + total.to_bytes(8, 'big', signed=False)
    else:
        yield b'x01' + int(0).to_bytes(8, 'big', signed=False)

    with pbar:
        _total_size = 0
        pbar.start_task(t)
        for doc in docs:
            doc_bytes = doc.to_bytes(protocol=protocol, compress=compress)
            len_doc_as_bytes = len(doc_bytes).to_bytes(4, 'big', signed=False)
            all_bytes = len_doc_as_bytes + doc_bytes

            yield all_bytes

            _total_size += len(all_bytes)
            pbar.update(
                t,
                advance=1,
                total_size=str(filesize.decimal(_total_size)),
            )
