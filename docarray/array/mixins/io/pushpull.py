import json
import os
import os.path
import warnings
from pathlib import Path
from typing import Dict, Type, TYPE_CHECKING, List, Optional, Any

import hubble
from hubble import Client as HubbleClient
from hubble.client.endpoints import EndpointsV2

from docarray.helper import (
    __cache_path__,
    _get_array_info,
    get_full_version,
)

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import T


def _get_length_from_summary(summary: List[Dict]) -> Optional[int]:
    """Get the length from summary."""
    for item in summary:
        if 'Length' == item['name']:
            return item['value']


class PushPullMixin:
    """Transmitting :class:`DocumentArray` via Jina Cloud Service"""

    _max_bytes = 4 * 1024 * 1024 * 1024

    @staticmethod
    @hubble.login_required
    def cloud_list(show_table: bool = False) -> List[str]:
        """List all available arrays in the cloud.

        :param show_table: if true, show the table of the arrays.
        :returns: List of available DocumentArray's names.
        """
        from rich import print

        result = []
        from rich.table import Table
        from rich import box

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
    def cloud_delete(name: str) -> None:
        """
        Delete a DocumentArray from the cloud.
        :param name: the name of the DocumentArray to delete.
        """
        HubbleClient(jsonify=True).delete_artifact(name=name)

    def _get_raw_summary(self) -> List[Dict[str, Any]]:
        (
            is_homo,
            _nested_in,
            _nested_items,
            attr_counter,
            all_attrs_names,
        ) = _get_array_info(self)

        items = [
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
                value=is_homo,
                description='Whether all documents are of the same structure, attributes',
            ),
            dict(
                name='Common Attributes',
                value=list(attr_counter.items())[0][0] if attr_counter else None,
                description='The common attributes of all documents',
            ),
            dict(
                name='Has nested Documents in',
                value=tuple(_nested_in),
                description='The field that contains nested Documents',
            ),
            dict(
                name='Multimodal dataclass',
                value=all(d.is_multimodal for d in self),
                description='Whether all documents are multimodal',
            ),
            dict(
                name='Subindices', value=tuple(getattr(self, '_subindices', {}).keys())
            ),
        ]

        items.append(
            dict(
                name='Inspect attributes',
                value=_nested_items,
                description='Quick overview of attributes of all documents',
            )
        )

        storage_infos = self._get_storage_infos()
        _nested_items = []
        if storage_infos:
            for k, v in storage_infos.items():
                _nested_items.append(dict(name=k, value=v))
        items.append(
            dict(
                name='Storage backend',
                value=_nested_items,
                description='Quick overview of the Document Store',
            )
        )

        return items

    @hubble.login_required
    def push(
        self,
        name: str,
        show_progress: bool = False,
        public: bool = True,
        branding: Optional[Dict] = None,
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
        :param branding: a dict of branding information to be sent to Jina Cloud. {"icon": "emoji", "background": "#fff"}
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
                'metaData': json.dumps(
                    {
                        'summary': self._get_raw_summary(),
                        'branding': branding,
                        'version': get_full_version(),
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
            response.raise_for_status()

    @classmethod
    @hubble.login_required
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
            r.raise_for_status()

            _da_len = int(r.headers['Content-length'])

            from docarray.array.mixins.io.binary import LazyRequestReader

            _source = LazyRequestReader(r)

            cache_file = f'{__cache_path__}/{name.replace("/", "_")}.da'
            if local_cache and os.path.exists(cache_file):
                _cache_len = os.path.getsize(cache_file)
                if _cache_len == _da_len:
                    _source = cache_file

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
                with open(cache_file, 'wb') as fp:
                    fp.write(_source.content)

            return r

    cloud_push = push
    cloud_pull = pull
