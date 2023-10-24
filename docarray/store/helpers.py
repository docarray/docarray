# It is usually a bad idea to have a helper file because it means we don't know where to put the code (or haven't put much thought into it).
# With that said, rules are meant to be broken, we will live with this for now.
from contextlib import nullcontext
from typing import Dict, Iterable, Iterator, NoReturn, Optional, Sequence, Type, TypeVar

from rich import filesize
from typing_extensions import TYPE_CHECKING, Protocol

from docarray.utils._internal.misc import ProtocolType
from docarray.utils._internal.progress_bar import _get_progressbar

if TYPE_CHECKING:
    from pathlib import Path

    import requests

CACHING_REQUEST_READER_CHUNK_SIZE = 2**20


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
        super().__init__(
            r.iter_content(chunk_size=CACHING_REQUEST_READER_CHUNK_SIZE), cache_path
        )


def raise_req_error(resp: 'requests.Response') -> NoReturn:
    """Definitely raise an error from a response."""
    resp.raise_for_status()
    raise ValueError(f'Unexpected response status: {resp.status_code}')


T_Elem = TypeVar('T_Elem')


class Streamable(Protocol):
    """A protocol for streamable objects."""

    def to_bytes(self, protocol: ProtocolType, compress: Optional[str]) -> bytes:
        ...

    @classmethod
    def from_bytes(
        cls: Type[T_Elem], bytes: bytes, protocol: ProtocolType, compress: Optional[str]
    ) -> 'T_Elem':
        ...


class ReadableBytes(Protocol):
    def read(self, size: int = -1) -> bytes:
        ...

    def close(self):
        ...


def _to_binary_stream(
    iterator: Iterator['Streamable'],
    total: Optional[int] = None,
    protocol: ProtocolType = 'protobuf',
    compress: Optional[str] = None,
    show_progress: bool = False,
) -> Iterator[bytes]:
    if show_progress:
        pbar, t = _get_progressbar(
            'Serializing', disable=not show_progress, total=total
        )
    else:
        pbar = nullcontext()

    with pbar:
        if show_progress:
            _total_size = 0
            count = 0
            pbar.start_task(t)
        for item in iterator:
            item_bytes = item.to_bytes(protocol=protocol, compress=compress)
            len_item_as_bytes = len(item_bytes).to_bytes(4, 'big', signed=False)
            all_bytes = len_item_as_bytes + item_bytes
            yield all_bytes

            if show_progress:
                _total_size += len(all_bytes)
                count += 1
                pbar.update(t, advance=1, total_size=str(filesize.decimal(_total_size)))

    yield int(0).to_bytes(4, 'big', signed=False)


T = TypeVar('T', bound=Streamable)


def _from_binary_stream(
    cls: Type[T],
    stream: ReadableBytes,
    total: Optional[int] = None,
    protocol: ProtocolType = 'protobuf',
    compress: Optional[str] = None,
    show_progress: bool = False,
) -> Iterator['T']:
    try:
        if show_progress:
            pbar, t = _get_progressbar(
                'Deserializing', disable=not show_progress, total=total
            )
        else:
            pbar = nullcontext()

        with pbar:
            if show_progress:
                _total_size = 0
                pbar.start_task(t)
            while True:
                len_bytes = stream.read(4)
                if len(len_bytes) < 4:
                    raise ValueError('Unexpected end of stream')
                len_item = int.from_bytes(len_bytes, 'big', signed=False)
                if len_item == 0:
                    break
                item_bytes = stream.read(len_item)
                if len(item_bytes) < len_item:
                    raise ValueError('Unexpected end of stream')
                item = cls.from_bytes(item_bytes, protocol=protocol, compress=compress)

                yield item

                if show_progress:
                    _total_size += len_item + 4
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )
    finally:
        stream.close()
