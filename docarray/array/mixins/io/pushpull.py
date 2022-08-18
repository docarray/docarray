import os
import warnings
from pathlib import Path
from typing import Dict, Type, TYPE_CHECKING, Optional

from docarray.helper import get_request_header, __cache_path__

if TYPE_CHECKING:
    from docarray.typing import T


class PushPullMixin:
    """Transmitting :class:`DocumentArray` via Jina Cloud Service"""

    _max_bytes = 4 * 1024 * 1024 * 1024

    def push(
        self,
        name: str,
        show_progress: bool = False,
        public: bool = True,
    ) -> Dict:
        """Push this DocumentArray object to Jina Cloud which can be later retrieved via :meth:`.push`

        .. note::
            - Push with the same ``name`` will override the existing content.
            - Kinda like a public clipboard where everyone can override anyone's content.
              So to make your content survive longer, you may want to use longer & more complicated name.
            - The lifetime of the content is not promised atm, could be a day, could be a week. Do not use it for
              persistence. Only use this full temporary transmission/storage/clipboard.

        :param name: a name that later can be used for retrieve this :class:`DocumentArray`.
        :param show_progress: if to show a progress bar on pulling
        :param public: by default anyone can pull a DocumentArray if they know its name.
            Setting this to False will allow only the creator to pull it. This feature of course you to login first.
        """
        import requests

        delimiter = os.urandom(32)

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(
            {
                'file': (
                    'DocumentArray',
                    delimiter,
                ),
                'name': name,
                'type': 'documentArray',
                'public': public,
            }
        )

        headers = {'Content-Type': ctype, **get_request_header()}
        import hubble

        auth_token = hubble.get_token()
        if auth_token:
            headers['Authorization'] = f'token {auth_token}'

        if not public and not auth_token:
            warnings.warn(
                'You are not logged in, and `public=False` if only for logged in users. To login, use `jina auth login`.'
            )

        _head, _tail = data.split(delimiter)
        _head += self._stream_header
        from rich import filesize
        from docarray.array.mixins.io.pbar import get_progressbar

        pbar, t = get_progressbar('Pushing', disable=not show_progress, total=len(self))

        def gen():
            total_size = 0

            pbar.start_task(t)

            yield _head

            def _get_chunk(_batch):
                return b''.join(
                    d._to_stream_bytes(protocol='protobuf', compress='gzip')
                    for d in _batch
                ), len(_batch)

            for chunk, num_doc_in_chunk in self.map_batch(_get_chunk, batch_size=32):
                total_size += len(chunk)
                if total_size > self._max_bytes:
                    warnings.warn(
                        f'DocumentArray is too big. The pushed DocumentArray might be chopped off.'
                    )
                    break
                yield chunk
                pbar.update(
                    t,
                    advance=num_doc_in_chunk,
                    total_size=str(filesize.decimal(total_size)),
                )
            yield _tail

        with pbar:
            from hubble import Client
            from hubble.client.endpoints import EndpointsV2

            response = requests.post(
                Client()._base_url + EndpointsV2.upload_artifact,
                data=gen(),
                headers=headers,
            )

        if response.ok:
            return response.json()['data']
        else:
            response.raise_for_status()

    @classmethod
    def pull(
        cls: Type['T'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = True,
        *args,
        **kwargs,
    ) -> 'T':
        """Pulling a :class:`DocumentArray` from Jina Cloud Service to local.

        :param name: the upload name set during :meth:`.push`
        :param show_progress: if to show a progress bar on pulling
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: a :class:`DocumentArray` object
        """

        import requests

        headers = {}

        import hubble

        auth_token = hubble.get_token()

        if auth_token:
            headers['Authorization'] = f'token {auth_token}'

        from hubble import Client
        from hubble.client.endpoints import EndpointsV2

        url = Client()._base_url + EndpointsV2.download_artifact + f'?name={name}'
        response = requests.get(url, headers=headers)

        if response.ok:
            url = response.json()['data']['download']
        else:
            response.raise_for_status()

        with requests.get(
            url,
            stream=True,
            headers=get_request_header(),
        ) as r:
            r.raise_for_status()

            _da_len = int(r.headers['Content-length'])

            from docarray.array.mixins.io.binary import LazyRequestReader

            _source = LazyRequestReader(r)
            if local_cache and os.path.exists(f'{__cache_path__}/{name}'):
                _cache_len = os.path.getsize(f'{__cache_path__}/{name}')
                if _cache_len == _da_len:
                    _source = f'{__cache_path__}/{name}'

            r = cls.load_binary(
                _source,
                protocol='protobuf',
                compress='gzip',
                _show_progress=show_progress,
                *args,
                **kwargs,
            )

            if isinstance(_source, LazyRequestReader) and local_cache:
                Path(__cache_path__).mkdir(parents=True, exist_ok=True)
                with open(f'{__cache_path__}/{name}', 'wb') as fp:
                    fp.write(_source.content)

            return r
