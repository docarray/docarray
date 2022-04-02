from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .. import ParallelMixin, GroupMixin
from ....helper import protocol_and_compress_from_file_path

if TYPE_CHECKING:
    from docarray import Document, DocumentArray


class DocumentArrayLoader(ParallelMixin, GroupMixin):
    def __init__(
        self,
        path: Union[str, Path],
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ):
        self._show_progress = show_progress
        self._filename = path
        self._protocol, self._compress = protocol_and_compress_from_file_path(
            path, protocol, compress
        )

        with open(path, 'rb') as f:
            version_numdocs_lendoc0 = f.read(9)
            # 8 bytes (uint64)
            self._len = int.from_bytes(
                version_numdocs_lendoc0[1:9], 'big', signed=False
            )
        self._iter = iter(self)

    def __iter__(self):
        from docarray import Document

        from ..io.pbar import get_progressbar
        from rich import filesize

        with open(self._filename, 'rb') as f:
            f.read(9)

            pbar, t = get_progressbar(
                'Deserializing', disable=not self._show_progress, total=self._len
            )

            with pbar:
                _total_size = 0
                pbar.start_task(t)
                for _ in range(self._len):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        f.read(4), 'big', signed=False
                    )
                    _total_size += len_current_doc_in_bytes
                    yield Document.from_bytes(
                        f.read(len_current_doc_in_bytes),
                        protocol=self._protocol,
                        compress=self._compress,
                    )
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )

    def __len__(self):
        return self._len

    def __getitem__(self, item: list) -> 'DocumentArray':
        from docarray import DocumentArray

        da = DocumentArray()
        for _ in item:
            da.append(next(self._iter))
        return da
