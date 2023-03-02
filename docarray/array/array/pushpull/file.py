from pathlib import Path
from typing import Dict, Iterator, List, Optional, Type

from typing_extensions import TYPE_CHECKING

from docarray.array.array.pushpull import __cache_path__
from docarray.array.array.pushpull.helpers import _from_binary_stream, _to_binary_stream

if TYPE_CHECKING:
    from docarray import BaseDocument, DocumentArray


class PushPullFile:
    @staticmethod
    def _abs_filepath(name: str) -> Path:
        """Resolve a name to an absolute path.
        If it is not a path, the cache directoty is prepended.
        If it is a path, it is resolved to an absolute path.
        """
        if not (name.startswith('/') or name.startswith('~') or name.startswith('.')):
            name = str(__cache_path__ / name)
        if name.startswith('~'):
            name = str(Path.home() / name[2:])
        return Path(name).resolve()

    @staticmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        """List all DocumentArrays in a directory.

        :param namespace: The directory to list.
        :param show_table: If True, print a table of the files in the directory.
        :return: A list of the names of the DocumentArrays in the directory.
        """
        namespace_dir = PushPullFile._abs_filepath(namespace)
        if not namespace_dir.exists():
            raise FileNotFoundError(f'Directory {namespace} does not exist')
        da_files = [dafile for dafile in namespace_dir.glob('*.da')]

        # TODO: Make nicer
        if show_table:
            print('Name', 'Size', 'Last Modified', sep='\t\t')
            print(*map(lambda x: f'{x.stem}\t{x.stat().st_size}', da_files), sep='\n')

        return [dafile.stem for dafile in da_files]

    @staticmethod
    def delete(name: str, missing_ok: bool = False) -> bool:
        """Delete a DocumentArray from the local filesystem.

        :param name: The name of the DocumentArray to delete.
        :param missing_ok: If True, do not raise an exception if the file does not exist. Defaults to False.
        :return: True if the file was deleted, False if it did not exist.
        """
        path = PushPullFile._abs_filepath(name)
        try:
            path.with_suffix('.da').unlink()
            return True
        except FileNotFoundError:
            if not missing_ok:
                raise
        return False

    @staticmethod
    def push(
        da: 'DocumentArray',
        url: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        return PushPullFile.push_stream(iter(da), url, public, show_progress, branding)

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDocument'],
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        source = _to_binary_stream(
            docs, protocol='protobuf', compress='gzip', show_progress=show_progress
        )
        path = PushPullFile._abs_filepath(name).with_suffix('.da')
        with open(path, 'wb') as f:
            while True:
                try:
                    f.write(next(source))
                except StopIteration:
                    break
        return {}

    @staticmethod
    def pull(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocumentArray':
        return cls(
            _from_binary_stream(
                cls.document_type,
                open(name + '.da', 'rb'),
                protocol='protobuf',
                compress='gzip',
                show_progress=show_progress,
            )
        )

    @staticmethod
    def pull_stream(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDocument']:
        path = PushPullFile._abs_filepath(name).with_suffix('.da')
        source = open(path, 'rb')
        return _from_binary_stream(
            cls.document_type,
            source,
            protocol='protobuf',
            compress='gzip',
            show_progress=show_progress,
        )
