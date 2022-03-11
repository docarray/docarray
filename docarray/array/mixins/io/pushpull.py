import json
import os
import warnings
from functools import lru_cache
from typing import Type, TYPE_CHECKING
from urllib.request import Request, urlopen

from ....helper import get_request_header

if TYPE_CHECKING:
    from ....types import T


@lru_cache()
def _get_cloud_api() -> str:
    """Get Cloud Api for transmiting data to the cloud.

    :raises RuntimeError: Encounter error when fetching the cloud Api Url.
    :return: Cloud Api Url
    """
    if 'JINA_HUBBLE_REGISTRY' in os.environ:
        u = os.environ['JINA_HUBBLE_REGISTRY']
    else:
        try:
            req = Request(
                'https://api.jina.ai/hub/hubble.json',
                headers={'User-Agent': 'Mozilla/5.0'},
            )
            with urlopen(req) as resp:
                u = json.load(resp)['url']
        except Exception as ex:
            raise RuntimeError(
                f'Can not fetch Cloud API address from {req.full_url}'
            ) from ex

    return u


class PushPullMixin:
    """Transmitting :class:`DocumentArray` via Jina Cloud Service"""

    _max_bytes = 4 * 1024 * 1024 * 1024

    def push(self, token: str, show_progress: bool = False) -> None:
        """Push this DocumentArray object to Jina Cloud which can be later retrieved via :meth:`.push`

        .. note::
            - Push with the same ``token`` will override the existing content.
            - Kinda like a public clipboard where everyone can override anyone's content.
              So to make your content survive longer, you may want to use longer & more complicated token.
            - The lifetime of the content is not promised atm, could be a day, could be a week. Do not use it for
              persistence. Only use this full temporary transmission/storage/clipboard.

        :param token: a key that later can be used for retrieve this :class:`DocumentArray`.
        :param show_progress: if to show a progress bar on pulling
        """
        import requests

        delimiter = os.urandom(32)

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(
            {
                'file': (
                    'DocumentArray',
                    delimiter,
                ),
                'token': token,
            }
        )

        headers = {'Content-Type': ctype, **get_request_header()}

        _head, _tail = data.split(delimiter)
        from rich.progress import track

        def gen():
            total_size = 0

            for idx, d in enumerate(
                track(self, description='Pushing', disable=not show_progress)
            ):
                chunk = b''
                if idx == 0:
                    chunk += _head
                    chunk += self._stream_header
                if idx < len(self):
                    chunk += d._to_stream_bytes(protocol='protobuf', compress='gzip')
                    total_size += len(chunk)
                    if total_size > self._max_bytes:
                        warnings.warn(
                            f'DocumentArray is too big. Only first {idx} Documents are pushed'
                        )
                        break
                    yield chunk
            yield _tail

        res = requests.post(
            f'{_get_cloud_api()}/v2/rpc/da.push', data=gen(), headers=headers
        )

        if res.status_code != 200:
            json_res = res.json()
            raise RuntimeError(
                json_res.get('message', 'Failed to push DocumentArray to Jina Cloud'),
                f'Status code: {res.status_code}',
            )

    @classmethod
    def pull(
        cls: Type['T'],
        token: str,
        show_progress: bool = False,
        local_cache: bool = False,
        *args,
        **kwargs,
    ) -> 'T':
        """Pulling a :class:`DocumentArray` from Jina Cloud Service to local.

        :param token: the upload token set during :meth:`.push`
        :param show_progress: if to show a progress bar on pulling
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: a :class:`DocumentArray` object
        """

        import requests

        url = f'{_get_cloud_api()}/v2/rpc/da.pull?token={token}'
        response = requests.get(url)

        url = response.json()['data']['download']

        with requests.get(
            url,
            stream=True,
            headers=get_request_header(),
        ) as r:
            r.raise_for_status()

            _da_len = int(r.headers['Content-length'])

            from .binary import LazyRequestReader

            _source = LazyRequestReader(r)
            if local_cache and os.path.exists(f'.cache/{token}'):
                _cache_len = os.path.getsize(f'.cache/{token}')
                if _cache_len == _da_len:
                    _source = f'.cache/{token}'

            r = cls.load_binary(
                _source,
                protocol='protobuf',
                compress='gzip',
                _show_progress=show_progress,
                *args,
                **kwargs,
            )

            if isinstance(_source, LazyRequestReader) and local_cache:
                os.makedirs('.cache', exist_ok=True)
                with open(f'.cache/{token}', 'wb') as fp:
                    fp.write(_source.content)

            return r


def _get_progressbar(show_progress):
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    return Progress(
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=True,
        disable=not show_progress,
    )
