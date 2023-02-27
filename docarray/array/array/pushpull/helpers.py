# It is usually a bad idea to have a helper file because it means we don't know where to put the code (or haven't put much thought into it).
# With that said, rules are meant to be broken, we will live with this for now.
from typing import Dict, Iterable, NoReturn, Optional, Sequence

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import requests


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


class _BufferedCachingRequestReader:
    """A buffered reader for requests.Response that writes to a cache file while reading."""

    def __init__(self, r: 'requests.Response', cache_path: Optional['Path'] = None):
        self._data = r.iter_content(chunk_size=1024 * 1024)
        self._chunk: bytes = b''
        self._seek = 0
        self._chunk_len = 0

        self._cache = open(cache_path, 'wb') if cache_path else None

    def read(self, size: int) -> bytes:
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


def raise_req_error(resp: 'requests.Response') -> NoReturn:
    """Definitely raise an error from a response."""
    resp.raise_for_status()
    raise ValueError(f'Unexpected response status: {resp.status_code}')
