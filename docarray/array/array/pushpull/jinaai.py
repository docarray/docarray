import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import hubble
from hubble import Client as HubbleClient
from hubble.client.endpoints import EndpointsV2

from docarray.array.array.pushpull import PushPullLike, __cache_path__
from docarray.array.array.pushpull.helpers import (
    _BufferedCachingRequestReader,
    get_version_info,
    raise_req_error,
)

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


class PushPullJAC(PushPullLike):
    """Class to push and pull DocumentArray to and from Jina AI Cloud."""

    @staticmethod
    @hubble.login_required
    def list(namespace: str, show_table: bool = False) -> List[str]:
        """List all available arrays in the cloud.

        :param show_table: if true, show the table of the arrays.
        :returns: List of available DocumentArray's names.
        """
        # TODO: Use the namespace
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
        # TODO: Add namespace?
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
                        'version': get_version_info(),
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
            raise_req_error(response)

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
            from contextlib import nullcontext

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
