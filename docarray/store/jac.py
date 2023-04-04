import json
import logging
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from docarray.store.abstract_doc_store import AbstractDocStore
from docarray.store.helpers import (
    _BufferedCachingRequestReader,
    get_version_info,
    raise_req_error,
)
from docarray.utils._internal.cache import _get_cache_path
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:  # pragma: no cover
    import io

    from docarray import BaseDoc, DocArray

if TYPE_CHECKING:
    import hubble
    from hubble import Client as HubbleClient
    from hubble.client.endpoints import EndpointsV2
else:
    hubble = import_library('hubble', raise_error=True)
    HubbleClient = hubble.Client
    EndpointsV2 = hubble.client.endpoints.EndpointsV2


def _get_length_from_summary(summary: List[Dict]) -> Optional[int]:
    """Get the length from summary."""
    for item in summary:
        if 'Length' == item['name']:
            return item['value']
    raise ValueError('Length not found in summary')


def _get_raw_summary(self: 'DocArray') -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = [
        dict(
            name='Type',
            value=self.__class__.__name__,
            description='The type of the DocArray',
        ),
        dict(
            name='Length',
            value=len(self),
            description='The length of the DocArray',
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


SelfJACDocStore = TypeVar('SelfJACDocStore', bound='JACDocStore')


class JACDocStore(AbstractDocStore):
    """Class to push and pull DocArray to and from Jina AI Cloud."""

    @staticmethod
    @hubble.login_required
    def list(namespace: str = '', show_table: bool = False) -> List[str]:
        """List all available arrays in the cloud.

        :param namespace: Not supported for Jina AI Cloud.
        :param show_table: if true, show the table of the arrays.
        :returns: List of available DocArray's names.
        """
        if len(namespace) > 0:
            logging.warning('Namespace is not supported for Jina AI Cloud.')
        from rich import print

        result = []
        from rich import box
        from rich.table import Table

        resp = HubbleClient(jsonify=True).list_artifacts(
            filter={'type': 'DocArray'}, sort={'createdAt': 1}
        )

        table = Table(
            title=f'You have {resp["meta"]["total"]} DocArray on the cloud',
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
    def delete(name: str, missing_ok: bool = True) -> bool:
        """
        Delete a DocArray from the cloud.
        :param name: the name of the DocArray to delete.
        :param missing_ok: if true, do not raise an error if the DocArray does not exist.
        :return: True if the DocArray was deleted, False if it did not exist.
        """
        try:
            HubbleClient(jsonify=True).delete_artifact(name=name)
        except hubble.excepts.RequestedEntityNotFoundError:
            if missing_ok:
                return False
            else:
                raise
        return True

    @staticmethod
    @hubble.login_required
    def push(
        da: 'DocArray',
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push this DocArray object to Jina AI Cloud

        .. note::
            - Push with the same ``name`` will override the existing content.
            - Kinda like a public clipboard where everyone can override anyone's content.
              So to make your content survive longer, you may want to use longer & more complicated name.
            - The lifetime of the content is not promised atm, could be a day, could be a week. Do not use it for
              persistence. Only use this full temporary transmission/storage/clipboard.

        :param name: A name that can later be used to retrieve this :class:`DocArray`.
        :param public: By default, anyone can pull a DocArray if they know its name.
            Setting this to false will restrict access to only the creator.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: A dictionary of branding information to be sent to Jina Cloud. e.g. {"icon": "emoji", "background": "#fff"}
        """
        import requests
        import urllib3

        delimiter = os.urandom(32)

        data, ctype = urllib3.filepost.encode_multipart_formdata(
            {
                'file': (
                    'DocArray',
                    delimiter,
                ),
                'name': name,
                'type': 'DocArray',
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

    @classmethod
    @hubble.login_required
    def push_stream(
        cls: Type[SelfJACDocStore],
        docs: Iterator['BaseDoc'],
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push a stream of documents to Jina AI Cloud

        .. note::
            - Push with the same ``name`` will override the existing content.
            - Kinda like a public clipboard where everyone can override anyone's content.
              So to make your content survive longer, you may want to use longer & more complicated name.
            - The lifetime of the content is not promised atm, could be a day, could be a week. Do not use it for
              persistence. Only use this full temporary transmission/storage/clipboard.

        :param name: A name that can later be used to retrieve this :class:`DocArray`.
        :param public: By default, anyone can pull a DocArray if they know its name.
            Setting this to false will restrict access to only the creator.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: A dictionary of branding information to be sent to Jina Cloud. e.g. {"icon": "emoji", "background": "#fff"}
        """
        from docarray import DocArray

        # This is a temporary solution to push a stream of documents
        # The memory footprint is not ideal
        # But it must be done this way for now because Hubble expects to know the length of the DocArray
        # before it starts receiving the documents
        first_doc = next(docs)
        da = DocArray[first_doc.__class__]([first_doc])  # type: ignore
        for doc in docs:
            da.append(doc)
        return cls.push(da, name, public, show_progress, branding)

    @staticmethod
    @hubble.login_required
    def pull(
        cls: Type['DocArray'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = True,
    ) -> 'DocArray':
        """Pull a :class:`DocArray` from Jina AI Cloud to local.

        :param name: the upload name set during :meth:`.push`
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocArray to local folder
        :return: a :class:`DocArray` object
        """
        from docarray import DocArray

        return DocArray[cls.document_type](  # type: ignore
            JACDocStore.pull_stream(cls, name, show_progress, local_cache)
        )

    @staticmethod
    @hubble.login_required
    def pull_stream(
        cls: Type['DocArray'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> Iterator['BaseDoc']:
        """Pull a :class:`DocArray` from Jina AI Cloud to local.

        :param name: the upload name set during :meth:`.push`
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocArray to local folder
        :return: An iterator of Documents
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

            r.raise_for_status()
            save_name = name.replace('/', '_')

            tmp_cache_file = Path(f'/tmp/{save_name}.da')
            _source: Union[
                _BufferedCachingRequestReader, io.BufferedReader
            ] = _BufferedCachingRequestReader(r, tmp_cache_file)

            cache_file = _get_cache_path() / f'{save_name}.da'
            if local_cache and cache_file.exists():
                _cache_len = cache_file.stat().st_size
                if _cache_len == int(r.headers['Content-length']):
                    if show_progress:
                        print(f'Loading from local cache {cache_file}')
                    _source = open(cache_file, 'rb')
                    r.close()

            docs = cls._load_binary_stream(
                nullcontext(_source),  # type: ignore
                protocol='protobuf',
                compress='gzip',
                show_progress=show_progress,
            )
            try:
                while True:
                    yield next(docs)
            except StopIteration:
                pass

            if local_cache:
                if isinstance(_source, _BufferedCachingRequestReader):
                    Path(_get_cache_path()).mkdir(parents=True, exist_ok=True)
                    tmp_cache_file.rename(cache_file)
                else:
                    _source.close()
