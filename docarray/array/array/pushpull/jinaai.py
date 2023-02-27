import json
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Sequence,
    Type,
    Union,
)

import hubble
import requests
from hubble import Client as HubbleClient
from hubble.client.endpoints import EndpointsV2

from docarray.array.array.pushpull import PushPullLike, __cache_path__

if TYPE_CHECKING:  # pragma: no cover
    import io

    from docarray import DocumentArray


def _get_length_from_summary(summary: List[Dict]) -> Optional[int]:
    """Get the length from summary."""
    for item in summary:
        if 'Length' == item['name']:
            return item['value']
    raise ValueError('Length not found in summary')


def _get_raw_summary(self: 'DocumentArray') -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = [
        dict(
            name='Type',
            value=self.__class__.__name__,
            description='The type of the DocumentArray',
        ),
        dict(
            name='Length',
            value=len(self),
            description='The length of the DocumentArray',
        ),
        dict(
            name='Homogenous Documents',
            value=True,
            description='Whether all documents are of the same structure, attributes',
        ),
        dict(
            name='Fields',
            value=tuple(self[0].__class__.__fields__.keys()),
            description='The fields of the Document',
        ),
        dict(
            name='Multimodal dataclass',
            value=True,
            description='Whether all documents are multimodal',
        ),
    ]

    return items


def _get_full_version() -> Dict:
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


## Parallels
def ibatch(iterable: Sequence, batch_size: int = 32) -> Iterable:
    """Get an iterator of batched items from Sequence."""
    seq_len = len(iterable)
    for offset in range(0, seq_len, batch_size):
        yield iterable[offset : min(offset + batch_size, seq_len)]


## Parallels


class _BufferedCachingRequestReader:
    """A buffered reader for requests.Response that writes to a cache file while reading."""

    def __init__(self, r: requests.Response, cache_path: Optional[Path] = None):
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


def _raise_req_error(resp: requests.Response) -> NoReturn:
    resp.raise_for_status()
    raise ValueError(f'Unexpected response status: {resp.status_code}')


class PushPullJAC(PushPullLike):
    """Class to push and pull DocumentArray to and from Jina AI Cloud."""

    _max_bytes = 4 * 2**30

    @staticmethod
    @hubble.login_required
    def list(show_table: bool = False) -> List[str]:
        """List all available arrays in the cloud.

        :param show_table: if true, show the table of the arrays.
        :returns: List of available DocumentArray's names.
        """
        from rich import print

        result = []
        from rich import box
        from rich.table import Table

        resp = HubbleClient(jsonify=True).list_artifacts(
            filter={'type': 'documentArray'}, sort={'createdAt': 1}
        )

        table = Table(
            title=f'You have {resp["meta"]["total"]} DocumentArray on the cloud',
            box=box.SIMPLE,
            highlight=True,
        )
        table.add_column('Name')
        table.add_column('Length')
        table.add_column('Access')
        table.add_column('Created at', justify='center')
        table.add_column('Updated at', justify='center')

        for da in resp['data']:
            result.append(da['name'])

            table.add_row(
                da['name'],
                str(_get_length_from_summary(da['metaData'].get('summary', []))),
                da['visibility'],
                da['createdAt'],
                da['updatedAt'],
            )

        if show_table:
            print(table)
        return result

    @staticmethod
    @hubble.login_required
    def delete(name: str) -> None:
        """
        Delete a DocumentArray from the cloud.
        :param name: the name of the DocumentArray to delete.
        """
        HubbleClient(jsonify=True).delete_artifact(name=name)

    @staticmethod
    @hubble.login_required
    def push(
        da: 'DocumentArray',
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push this DocumentArray object to Jina AI Cloud which can be later retrieved via :meth:`.push`

        .. note::
            - Push with the same ``name`` will override the existing content.
            - Kinda like a public clipboard where everyone can override anyone's content.
              So to make your content survive longer, you may want to use longer & more complicated name.
            - The lifetime of the content is not promised atm, could be a day, could be a week. Do not use it for
              persistence. Only use this full temporary transmission/storage/clipboard.

        :param name: A name that can later be used to retrieve this :class:`DocumentArray`.
        :param public: By default, anyone can pull a DocumentArray if they know its name.
            Setting this to false will restrict access to only the creator.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: A dictionary of branding information to be sent to Jina Cloud. {"icon": "emoji", "background": "#fff"}
        """
        import requests
        import urllib3

        delimiter = os.urandom(32)

        data, ctype = urllib3.filepost.encode_multipart_formdata(
            {
                'file': (
                    'DocumentArray',
                    delimiter,
                ),
                'name': name,
                'type': 'documentArray',
                'public': public,
                'metaData': json.dumps(
                    {
                        'summary': _get_raw_summary(da),
                        'branding': branding,
                        'version': _get_full_version(),
                    },
                    sort_keys=True,
                ),
            }
        )

        headers = {
            'Content-Type': ctype,
        }

        auth_token = hubble.get_token()
        if auth_token:
            headers['Authorization'] = f'token {auth_token}'

        _head, _tail = data.split(delimiter)

        def gen():
            yield _head
            binary_stream = da.to_binary_stream(
                protocol='protobuf', compress='gzip', show_progress=show_progress
            )
            while True:
                try:
                    yield next(binary_stream)
                except StopIteration:
                    break
            yield _tail

        response = requests.post(
            HubbleClient()._base_url + EndpointsV2.upload_artifact,
            data=gen(),
            headers=headers,
        )

        if response.ok:
            return response.json()['data']
        else:
            if response.status_code >= 400 and 'readableMessage' in response.json():
                response.reason = response.json()['readableMessage']
            _raise_req_error(response)

    @staticmethod
    @hubble.login_required
    def pull(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = True,
    ):
        """Pull a :class:`DocumentArray` from Jina AI Cloud to local.

        :param name: the upload name set during :meth:`.push`
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: a :class:`DocumentArray` object
        """
        import requests

        from docarray.base_document import AnyDocument

        if cls.document_type == AnyDocument:
            raise TypeError(
                'There is no document schema defined. '
                'Please specify the DocumentArray\'s Document type using `DocumentArray[MyDoc]`.'
            )

        headers = {}

        auth_token = hubble.get_token()

        if auth_token:
            headers['Authorization'] = f'token {auth_token}'

        url = HubbleClient()._base_url + EndpointsV2.download_artifact + f'?name={name}'
        response = requests.get(url, headers=headers)

        if response.ok:
            url = response.json()['data']['download']
        else:
            response.raise_for_status()

        with requests.get(
            url,
            stream=True,
        ) as r:
            from docarray import DocumentArray

            r.raise_for_status()
            save_name = name.replace('/', '_')

            tmp_cache_file = Path(f'/tmp/{save_name}.da')
            _source: Union[
                _BufferedCachingRequestReader, io.BufferedReader
            ] = _BufferedCachingRequestReader(r, tmp_cache_file)

            cache_file = __cache_path__ / f'{save_name}.da'
            if local_cache and cache_file.exists():
                _cache_len = cache_file.stat().st_size
                if _cache_len == int(r.headers['Content-length']):
                    if show_progress:
                        print(f'Loading from local cache {cache_file}')
                    _source = open(cache_file, 'rb')
                    r.close()
            from contextlib import nullcontext

            da = DocumentArray[cls.document_type](  # type: ignore
                cls._load_binary_stream(
                    nullcontext(_source),  # type: ignore
                    protocol='protobuf',
                    compress='gzip',
                    show_progress=show_progress,
                )
            )

            if local_cache:
                if isinstance(_source, _BufferedCachingRequestReader):
                    Path(__cache_path__).mkdir(parents=True, exist_ok=True)
                    tmp_cache_file.rename(cache_file)
                else:
                    _source.close()

        return da
