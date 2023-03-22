import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Type, TypeVar

from typing_extensions import TYPE_CHECKING

from docarray.remote.abstract_doc_store import AbstractDocStore
from docarray.remote.exceptions import ConcurrentPushException
from docarray.remote.helpers import _from_binary_stream, _to_binary_stream
from docarray.utils.cache import get_cache_path

if TYPE_CHECKING:
    from docarray import BaseDocument, DocumentArray

SelfFileDocStore = TypeVar('SelfFileDocStore', bound='FileDocStore')


class FileDocStore(AbstractDocStore):
    @staticmethod
    def _abs_filepath(name: str) -> Path:
        """Resolve a name to an absolute path.
        If it is not a path, the cache directoty is prepended.
        If it is a path, it is resolved to an absolute path.
        """
        if not (name.startswith('/') or name.startswith('~') or name.startswith('.')):
            name = str(get_cache_path() / name)
        if name.startswith('~'):
            name = str(Path.home() / name[2:])
        return Path(name).resolve()

    @classmethod
    def list(
        cls: Type[SelfFileDocStore], namespace: str, show_table: bool
    ) -> List[str]:
        """List all DocumentArrays in a directory.

        :param namespace: The directory to list.
        :param show_table: If True, print a table of the files in the directory.
        :return: A list of the names of the DocumentArrays in the directory.
        """
        namespace_dir = cls._abs_filepath(namespace)
        if not namespace_dir.exists():
            raise FileNotFoundError(f'Directory {namespace} does not exist')
        da_files = [dafile for dafile in namespace_dir.glob('*.da')]

        if show_table:
            from datetime import datetime

            from rich import box, filesize
            from rich.console import Console
            from rich.table import Table

            table = Table(
                title=f'You have {len(da_files)} DocumentArrays in file://{namespace_dir}',
                box=box.SIMPLE,
                highlight=True,
            )
            table.add_column('Name')
            table.add_column('Last Modified', justify='center')
            table.add_column('Size')

            for da_file in da_files:
                table.add_row(
                    da_file.stem,
                    str(datetime.fromtimestamp(int(da_file.stat().st_ctime))),
                    str(filesize.decimal(da_file.stat().st_size)),
                )

            Console().print(table)

        return [dafile.stem for dafile in da_files]

    @classmethod
    def delete(
        cls: Type[SelfFileDocStore], name: str, missing_ok: bool = False
    ) -> bool:
        """Delete a DocumentArray from the local filesystem.

        :param name: The name of the DocumentArray to delete.
        :param missing_ok: If True, do not raise an exception if the file does not exist. Defaults to False.
        :return: True if the file was deleted, False if it did not exist.
        """
        path = cls._abs_filepath(name)
        try:
            path.with_suffix('.da').unlink()
            return True
        except FileNotFoundError:
            if not missing_ok:
                raise
        return False

    @classmethod
    def push(
        cls: Type[SelfFileDocStore],
        da: 'DocumentArray',
        name: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        """Push this DocumentArray object to the specified file path.

        :param name: The file path to push to.
        :param public: Not used by the ``file`` protocol.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Not used by the ``file`` protocol.
        """
        return cls.push_stream(iter(da), name, public, show_progress, branding)

    @classmethod
    def push_stream(
        cls: Type[SelfFileDocStore],
        docs: Iterator['BaseDocument'],
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push a stream of documents to the specified file path.

        :param docs: a stream of documents
        :param name: The file path to push to.
        :param public: Not used by the ``file`` protocol.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Not used by the ``file`` protocol.
        """
        if branding is not None:
            logging.warning('branding is not supported for "file" protocol')

        source = _to_binary_stream(
            docs, protocol='protobuf', compress='gzip', show_progress=show_progress
        )
        path = cls._abs_filepath(name).with_suffix('.da.tmp')
        if path.exists():
            raise ConcurrentPushException(f'File {path} already exists.')
        with open(path, 'wb') as f:
            while True:
                try:
                    f.write(next(source))
                except StopIteration:
                    break
        path.rename(path.with_suffix(''))
        return {}

    @classmethod
    def pull(
        cls: Type[SelfFileDocStore],
        da_cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocumentArray':
        """Pull a :class:`DocumentArray` from the specified url.

        :param name: The file path to pull from.
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: a :class:`DocumentArray` object
        """

        return da_cls(
            cls.pull_stream(
                da_cls, name, show_progress=show_progress, local_cache=local_cache
            )
        )

    @classmethod
    def pull_stream(
        cls: Type[SelfFileDocStore],
        da_cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDocument']:
        """Pull a stream of Documents from the specified file.

        :param name: The file path to pull from.
        :param show_progress: if true, display a progress bar.
        :param local_cache: Not used by the ``file`` protocol.
        :return: Iterator of Documents
        """

        if local_cache:
            logging.warning('local_cache is not supported for "file" protocol')

        path = cls._abs_filepath(name).with_suffix('.da')
        source = open(path, 'rb')
        return _from_binary_stream(
            da_cls.document_type,
            source,
            protocol='protobuf',
            compress='gzip',
            show_progress=show_progress,
        )
