import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Type, TypeVar

from docarray.store.abstract_doc_store import AbstractDocStore
from docarray.store.helpers import _from_binary_stream, _to_binary_stream
from docarray.utils._internal.cache import _get_cache_path
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:  # pragma: no cover
    import boto3
    import botocore
    from smart_open import open

    from docarray import BaseDoc, DocList
else:
    open = import_library('smart_open', raise_error=True).open
    boto3 = import_library('boto3', raise_error=True)
    botocore = import_library('botocore', raise_error=True)

SelfS3DocStore = TypeVar('SelfS3DocStore', bound='S3DocStore')


class _BufferedCachingReader:
    """A buffered reader that writes to a cache file while reading."""

    def __init__(
        self, iter_bytes: io.BufferedReader, cache_path: Optional['Path'] = None
    ):
        self._data = iter_bytes
        self._cache = None
        if cache_path:
            self._cache_path = cache_path.with_suffix('.tmp')
            self._cache = open(self._cache_path, 'wb')
        self.closed = False

    def read(self, size: Optional[int] = -1) -> bytes:
        bytes = self._data.read(size)
        if self._cache:
            self._cache.write(bytes)
        return bytes

    def close(self):
        if not self.closed and self._cache:
            self._cache_path.rename(self._cache_path.with_suffix('.docs'))
            self._cache.close()


class S3DocStore(AbstractDocStore):
    """Class to push and pull [`DocList`][docarray.DocList] to and from S3."""

    @staticmethod
    def list(namespace: str, show_table: bool = False) -> List[str]:
        """List all [`DocList`s][docarray.DocList] in the specified bucket and namespace.

        :param namespace: The bucket and namespace to list. e.g. my_bucket/my_namespace
        :param show_table: If true, a rich table will be printed to the console.
        :return: A list of `DocList` names.
        """
        bucket, namespace = namespace.split('/', 1)
        s3 = boto3.resource('s3')
        s3_bucket = s3.Bucket(bucket)
        da_files = [
            obj
            for obj in s3_bucket.objects.all()
            if obj.key.startswith(namespace) and obj.key.endswith('.docs')
        ]
        da_names = [f.key.split('/')[-1].split('.')[0] for f in da_files]

        if show_table:
            from rich import box, filesize
            from rich.console import Console
            from rich.table import Table

            table = Table(
                title=f'You have {len(da_files)} DocLists in bucket s3://{bucket} under the namespace "{namespace}"',
                box=box.SIMPLE,
                highlight=True,
            )
            table.add_column('Name')
            table.add_column('Last Modified', justify='center')
            table.add_column('Size')

            for da_name, da_file in zip(da_names, da_files):
                table.add_row(
                    da_name,
                    str(da_file.last_modified),
                    str(filesize.decimal(da_file.size)),
                )

            Console().print(table)
        return da_names

    @staticmethod
    def delete(name: str, missing_ok: bool = True) -> bool:
        """Delete the [`DocList`][docarray.DocList] object at the specified bucket and key.

        :param name: The bucket and key to delete. e.g. my_bucket/my_key
        :param missing_ok: If true, no error will be raised if the object does not exist.
        :return: True if the object was deleted, False if it did not exist.
        """
        bucket, name = name.split('/', 1)
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, name + '.docs')
        try:
            object.load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                if missing_ok:
                    return False
                else:
                    raise ValueError(f'Object {name} does not exist')
            else:
                raise
        object.delete()
        return True

    @classmethod
    def push(
        cls: Type[SelfS3DocStore],
        docs: 'DocList',
        name: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push this [`DocList`][docarray.DocList] object to the specified bucket and key.

        :param docs: The `DocList` to push.
        :param name: The bucket and key to push to. e.g. my_bucket/my_key
        :param show_progress: If true, a progress bar will be displayed.
        """
        return cls.push_stream(iter(docs), name, show_progress)

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDoc'],
        name: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push a stream of documents to the specified bucket and key.

        :param docs: a stream of documents
        :param name: The bucket and key to push to. e.g. my_bucket/my_key
        :param show_progress: If true, a progress bar will be displayed.
        """
        bucket, name = name.split('/', 1)
        binary_stream = _to_binary_stream(
            docs, protocol='pickle', compress=None, show_progress=show_progress
        )

        # Upload to S3
        with open(
            f"s3://{bucket}/{name}.docs",
            'wb',
            compression='.gz',
            transport_params={'multipart_upload': False},
        ) as fout:
            while True:
                try:
                    fout.write(next(binary_stream))
                except StopIteration:
                    break

        return {}

    @classmethod
    def pull(
        cls: Type[SelfS3DocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> 'DocList':
        """Pull a [`DocList`][docarray.DocList] from the specified bucket and key.

        :param name: The bucket and key to pull from. e.g. my_bucket/my_key
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocList to local cache
        :return: a `DocList` object
        """
        docs = docs_cls(  # type: ignore
            cls.pull_stream(
                docs_cls, name, show_progress=show_progress, local_cache=local_cache
            )
        )
        return docs

    @classmethod
    def pull_stream(
        cls: Type[SelfS3DocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of Documents from the specified name.
        Name is expected to be in the format of bucket/key.

        :param name: The bucket and key to pull from. e.g. my_bucket/my_key
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocList to local cache
        :return: An iterator of Documents
        """

        bucket, name = name.split('/', 1)

        save_name = name.replace('/', '_')
        cache_path = _get_cache_path() / f'{save_name}.docs'

        source = _BufferedCachingReader(
            open(f"s3://{bucket}/{name}.docs", 'rb', compression='.gz'),
            cache_path=cache_path if local_cache else None,
        )

        if local_cache:
            if cache_path.exists():
                object_header = boto3.client('s3').head_object(
                    Bucket=bucket, Key=name + '.docs'
                )
                if cache_path.stat().st_size == object_header['ContentLength']:
                    logging.info(
                        f'Using cached file for {name} (size: {cache_path.stat().st_size})'
                    )
                    source = open(cache_path, 'rb')

        return _from_binary_stream(
            docs_cls.doc_type,
            source,
            protocol='pickle',
            compress=None,
            show_progress=show_progress,
        )
