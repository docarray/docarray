import io
import json
import os
from contextlib import nullcontext
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

        dict_data = self._get_dict_data(token, show_progress)

        progress = _get_progressbar(show_progress)
        task_id = progress.add_task('upload', start=False) if show_progress else None

        class BufferReader(io.BytesIO):
            def __init__(self, buf=b'', p_bar=None, task_id=None):
                super().__init__(buf)
                self._len = len(buf)
                self._p_bar = p_bar
                self._task_id = task_id
                if show_progress:
                    progress.update(task_id, total=self._len)
                    progress.start_task(task_id)

            def __len__(self):
                return self._len

            def read(self, n=-1):
                chunk = io.BytesIO.read(self, n)
                if self._p_bar:
                    self._p_bar.update(self._task_id, advance=len(chunk))
                return chunk

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(
            dict_data
        )

        headers = {'Content-Type': ctype, **get_request_header()}

        with progress as p_bar:
            body = BufferReader(data, p_bar, task_id)
            res = requests.post(
                f'{_get_cloud_api()}/v2/rpc/da.push', data=body, headers=headers
            )

            if res.status_code != 200:
                json_res = res.json()
                raise RuntimeError(
                    json_res.get(
                        'message', 'Failed to push DocumentArray to Jina Cloud'
                    ),
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

        progress = _get_progressbar(show_progress)

        url = response.json()['data']['download']

        with requests.get(
            url,
            stream=True,
            headers=get_request_header(),
        ) as r, progress:
            r.raise_for_status()

            _da_len = int(r.headers['Content-length'])

            if local_cache and os.path.exists(f'.cache/{token}'):
                _cache_len = os.path.getsize(f'.cache/{token}')
                if _cache_len == _da_len:
                    if show_progress:
                        progress.stop()

                    return cls.load_binary(
                        f'.cache/{token}',
                        protocol='protobuf',
                        compress='gzip',
                        _show_progress=show_progress,
                        *args,
                        **kwargs,
                    )

            if show_progress:
                task_id = progress.add_task('download', start=False)
                progress.update(task_id, total=int(_da_len))
            with io.BytesIO() as f:
                chunk_size = 8192
                if show_progress:
                    progress.start_task(task_id)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    if show_progress:
                        progress.update(task_id, advance=len(chunk))

                if local_cache:
                    os.makedirs('.cache', exist_ok=True)
                    with open(f'.cache/{token}', 'wb') as fp:
                        fp.write(f.getbuffer())

                if show_progress:
                    progress.stop()

                return cls.from_bytes(
                    f.getvalue(),
                    protocol='protobuf',
                    compress='gzip',
                    _show_progress=show_progress,
                    *args,
                    **kwargs,
                )

    def _get_dict_data(self, token, show_progress):
        _serialized = self.to_bytes(
            protocol='protobuf', compress='gzip', _show_progress=show_progress
        )
        if len(_serialized) > self._max_bytes:
            raise ValueError(
                f'DocumentArray is too big. '
                f'Size of the serialization {len(_serialized)} is larger than {self._max_bytes}.'
            )

        return {
            'file': (
                'DocumentArray',
                _serialized,
            ),
            'token': token,
        }


def _get_progressbar(show_progress):
    if show_progress:
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
        )
    else:
        return nullcontext()
