import base64
import csv
import io
import os
import pathlib
import pickle
from abc import abstractmethod
from contextlib import nullcontext
from io import StringIO, TextIOWrapper
from itertools import compress
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import orjson

from docarray.base_doc import AnyDoc, BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.helper import (
    _access_path_dict_to_nested_dict,
    _all_access_paths_valid,
    _dict_to_access_paths,
)
from docarray.utils._internal.compress import _decompress_bytes, _get_compress_ctx
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import pandas as pd

    from docarray.array.doc_vec.doc_vec import DocVec
    from docarray.array.doc_vec.io import IOMixinDocVec
    from docarray.proto import DocListProto
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='IOMixinDocList')
T_doc = TypeVar('T_doc', bound=BaseDoc)

ARRAY_PROTOCOLS = {'protobuf-array', 'pickle-array', 'json-array'}
SINGLE_PROTOCOLS = {'pickle', 'protobuf', 'json'}
ALLOWED_PROTOCOLS = ARRAY_PROTOCOLS.union(SINGLE_PROTOCOLS)
ALLOWED_COMPRESSIONS = {'lz4', 'bz2', 'lzma', 'zlib', 'gzip'}


def _protocol_and_compress_from_file_path(
    file_path: Union[pathlib.Path, str],
    default_protocol: Optional[str] = None,
    default_compress: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract protocol and compression algorithm from a string, use defaults if not found.
    :param file_path: path of a file.
    :param default_protocol: default serialization protocol used in case not found.
    :param default_compress: default compression method used in case not found.
    Examples:
    >>> _protocol_and_compress_from_file_path('./docarray_fashion_mnist.protobuf.gzip')
    ('protobuf', 'gzip')
    >>> _protocol_and_compress_from_file_path('/Documents/docarray_fashion_mnist.protobuf')
    ('protobuf', None)
    >>> _protocol_and_compress_from_file_path('/Documents/docarray_fashion_mnist.gzip')
    (None, gzip)
    """

    protocol = default_protocol
    compress = default_compress

    file_extensions = [e.replace('.', '') for e in pathlib.Path(file_path).suffixes]
    for extension in file_extensions:
        if extension in ALLOWED_PROTOCOLS:
            protocol = extension
        elif extension in ALLOWED_COMPRESSIONS:
            compress = extension

    return protocol, compress


class _LazyRequestReader:
    def __init__(self, r):
        self._data = r.iter_content(chunk_size=1024 * 1024)
        self.content = b''

    def __getitem__(self, item: slice):
        while len(self.content) < item.stop:
            try:
                self.content += next(self._data)
            except StopIteration:
                return self.content[item.start : -1 : item.step]
        return self.content[item]


class IOMixinDocList(Iterable[T_doc]):
    doc_type: Type[T_doc]

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __init__(
        self,
        docs: Optional[Iterable[BaseDoc]] = None,
    ):
        ...

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocListProto') -> T:
        """create a Document from a protobuf message
        :param pb_msg: The protobuf message from where to construct the DocList
        """
        return cls(cls.doc_type.from_protobuf(doc_proto) for doc_proto in pb_msg.docs)

    def to_protobuf(self) -> 'DocListProto':
        """Convert `DocList` into a Protobuf message"""
        from docarray.proto import DocListProto

        da_proto = DocListProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        return da_proto

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> T:
        """Deserialize bytes into a `DocList`.

        :param data: Bytes from which to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compression algorithm that was used to serialize between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized `DocList`
        """
        return cls._load_binary_all(
            file_ctx=nullcontext(data),
            protocol=protocol,
            compress=compress,
            show_progress=show_progress,
        )

    def _write_bytes(
        self,
        bf: BinaryIO,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> None:
        if protocol in ARRAY_PROTOCOLS:
            compress_ctx = _get_compress_ctx(compress)
        else:
            # delegate the compression to per-doc compression
            compress_ctx = None

        fc: ContextManager
        if compress_ctx is None:
            # if compress do not support streaming then postpone the compress
            # into the for-loop
            f, fc = bf, nullcontext()
        else:
            f = compress_ctx(bf)
            fc = f
            compress = None

        with fc:
            if protocol == 'protobuf-array':
                f.write(self.to_protobuf().SerializePartialToString())
            elif protocol == 'pickle-array':
                f.write(pickle.dumps(self))
            elif protocol == 'json-array':
                f.write(self.to_json())
            elif protocol in SINGLE_PROTOCOLS:
                f.write(
                    b''.join(
                        self._to_binary_stream(
                            protocol=protocol,
                            compress=compress,
                            show_progress=show_progress,
                        )
                    )
                )
            else:
                raise ValueError(
                    f'protocol={protocol} is not supported. Can be only {ALLOWED_PROTOCOLS}.'
                )

    def _to_binary_stream(
        self,
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> Iterator[bytes]:
        from rich import filesize

        if show_progress:
            from docarray.utils._internal.progress_bar import _get_progressbar

            pbar, t = _get_progressbar(
                'Serializing', disable=not show_progress, total=len(self)
            )
        else:
            from contextlib import nullcontext

            pbar = nullcontext()

        yield self._stream_header

        with pbar:
            if show_progress:
                _total_size = 0
                pbar.start_task(t)
            for doc in self:
                doc_bytes = doc.to_bytes(protocol=protocol, compress=compress)
                len_doc_as_bytes = len(doc_bytes).to_bytes(4, 'big', signed=False)
                all_bytes = len_doc_as_bytes + doc_bytes

                yield all_bytes

                if show_progress:
                    _total_size += len(all_bytes)
                    pbar.update(
                        t,
                        advance=1,
                        total_size=str(filesize.decimal(_total_size)),
                    )

    def to_bytes(
        self,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        file_ctx: Optional[BinaryIO] = None,
        show_progress: bool = False,
    ) -> Optional[bytes]:
        """Serialize itself into `bytes`.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between : `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param file_ctx: File or filename or serialized bytes where the data is stored.
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the binary serialization in bytes or None if file_ctx is passed where to store
        """

        with file_ctx or io.BytesIO() as bf:
            self._write_bytes(
                bf=bf,
                protocol=protocol,
                compress=compress,
                show_progress=show_progress,
            )
            if isinstance(bf, io.BytesIO):
                return bf.getvalue()

        return None

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> T:
        """Deserialize base64 strings into a `DocList`.

        :param data: Base64 string to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized `DocList`
        """
        return cls._load_binary_all(
            file_ctx=nullcontext(base64.b64decode(data)),
            protocol=protocol,
            compress=compress,
            show_progress=show_progress,
        )

    def to_base64(
        self,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> str:
        """Serialize itself into base64 encoded string.

        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the binary serialization in bytes or None if file_ctx is passed where to store
        """
        with io.BytesIO() as bf:
            self._write_bytes(
                bf=bf,
                compress=compress,
                protocol=protocol,
                show_progress=show_progress,
            )
            return base64.b64encode(bf.getvalue()).decode('utf-8')

    @classmethod
    def from_json(
        cls: Type[T],
        file: Union[str, bytes, bytearray],
    ) -> T:
        """Deserialize JSON strings or bytes into a `DocList`.

        :param file: JSON object from where to deserialize a `DocList`
        :return: the deserialized `DocList`
        """
        json_docs = orjson.loads(file)
        return cls([cls.doc_type(**v) for v in json_docs])

    def to_json(self) -> bytes:
        """Convert the object into JSON bytes. Can be loaded via `.from_json`.
        :return: JSON serialization of `DocList`
        """
        return orjson_dumps(self)

    @classmethod
    def from_csv(
        cls: Type['T'],
        file_path: str,
        encoding: str = 'utf-8',
        dialect: Union[str, csv.Dialect] = 'excel',
    ) -> 'T':
        """
        Load a DocList from a csv file following the schema defined in the
        [`.doc_type`][docarray.DocList] attribute.
        Every row of the csv file will be mapped to one document in the doc_list.
        The column names (defined in the first row) have to match the field names
        of the Document type.
        For nested fields use "__"-separated access paths, such as `'image__url'`.

        List-like fields (including field of type DocList) are not supported.

        :param file_path: path to csv file to load DocList from.
        :param encoding: encoding used to read the csv file. Defaults to 'utf-8'.
        :param dialect: defines separator and how to handle whitespaces etc.
            Can be a [`csv.Dialect`](https://docs.python.org/3/library/csv.html#csv.Dialect)
            instance or one string of:
            `'excel'` (for comma separated values),
            `'excel-tab'` (for tab separated values),
            `'unix'` (for csv file generated on UNIX systems).

        :return: `DocList` object
        """
        if cls.doc_type == AnyDoc or cls.doc_type == BaseDoc:
            raise TypeError(
                'There is no document schema defined. '
                f'Please specify the {cls}\'s Document type using `{cls}[MyDoc]`.'
            )

        if file_path.startswith('http'):
            import urllib.request

            with urllib.request.urlopen(file_path) as f:
                file = StringIO(f.read().decode(encoding))
                return cls._from_csv_file(file, dialect)
        else:
            with open(file_path, 'r', encoding=encoding) as fp:
                return cls._from_csv_file(fp, dialect)

    @classmethod
    def _from_csv_file(
        cls: Type['T'],
        file: Union[StringIO, TextIOWrapper],
        dialect: Union[str, csv.Dialect],
    ) -> 'T':

        rows = csv.DictReader(file, dialect=dialect)

        doc_type = cls.doc_type
        docs = []

        field_names: List[str] = (
            [] if rows.fieldnames is None else [str(f) for f in rows.fieldnames]
        )
        if field_names is None or len(field_names) == 0:
            raise TypeError("No field names are given.")

        valid_paths = _all_access_paths_valid(
            doc_type=doc_type, access_paths=field_names
        )
        if not all(valid_paths):
            raise ValueError(
                f'Column names do not match the schema of the DocList\'s '
                f'document type ({cls.doc_type.__name__}): '
                f'{list(compress(field_names, [not v for v in valid_paths]))}'
            )

        for access_path2val in rows:
            doc_dict: Dict[Any, Any] = _access_path_dict_to_nested_dict(access_path2val)
            docs.append(doc_type.parse_obj(doc_dict))

        return cls(docs)

    def to_csv(
        self, file_path: str, dialect: Union[str, csv.Dialect] = 'excel'
    ) -> None:
        """
        Save a `DocList` to a csv file.
        The field names will be stored in the first row. Each row corresponds to the
        information of one Document.
        Columns for nested fields will be named after the "__"-seperated access paths,
        such as `'image__url'` for `image.url`.

        :param file_path: path to a csv file.
        :param dialect: defines separator and how to handle whitespaces etc.
            Can be a [`csv.Dialect`](https://docs.python.org/3/library/csv.html#csv.Dialect)
            instance or one string of:
            `'excel'` (for comma separated values),
            `'excel-tab'` (for tab separated values),
            `'unix'` (for csv file generated on UNIX systems).

        """
        if self.doc_type == AnyDoc or self.doc_type == BaseDoc:
            raise TypeError(
                f'{type(self)} must be homogeneous to be converted to a csv.'
                'There is no document schema defined. '
                f'Please specify the {type(self)}\'s Document type using `{type(self)}[MyDoc]`.'
            )
        fields = self.doc_type._get_access_paths()

        with open(file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields, dialect=dialect)
            writer.writeheader()

            for doc in self:
                doc_dict = _dict_to_access_paths(doc.dict())
                writer.writerow(doc_dict)

    @classmethod
    def from_dataframe(cls: Type['T'], df: 'pd.DataFrame') -> 'T':
        """
        Load a `DocList` from a `pandas.DataFrame` following the schema
        defined in the [`.doc_type`][docarray.DocList] attribute.
        Every row of the dataframe will be mapped to one Document in the doc_list.
        The column names of the dataframe have to match the field names of the
        Document type.
        For nested fields use "__"-separated access paths as column names,
        such as `'image__url'`.

        List-like fields (including field of type DocList) are not supported.

        ---

        ```python
        import pandas as pd

        from docarray import BaseDoc, DocList


        class Person(BaseDoc):
            name: str
            follower: int


        df = pd.DataFrame(
            data=[['Maria', 12345], ['Jake', 54321]], columns=['name', 'follower']
        )

        docs = DocList[Person].from_dataframe(df)

        assert docs.name == ['Maria', 'Jake']
        assert docs.follower == [12345, 54321]
        ```

        ---

        :param df: `pandas.DataFrame` to extract Document's information from
        :return: `DocList` where each Document contains the information of one
            corresponding row of the `pandas.DataFrame`.
        """
        from docarray import DocList

        if cls.doc_type == AnyDoc or cls.doc_type == BaseDoc:
            raise TypeError(
                'There is no document schema defined. '
                f'Please specify the {cls}\'s Document type using `{cls}[MyDoc]`.'
            )

        doc_type = cls.doc_type
        docs = DocList.__class_getitem__(doc_type)()
        field_names = df.columns.tolist()

        if field_names is None or len(field_names) == 0:
            raise TypeError("No field names are given.")

        valid_paths = _all_access_paths_valid(
            doc_type=doc_type, access_paths=field_names
        )
        if not all(valid_paths):
            raise ValueError(
                f'Column names do not match the schema of the DocList\'s '
                f'document type ({cls.doc_type.__name__}): '
                f'{list(compress(field_names, [not v for v in valid_paths]))}'
            )

        for row in df.itertuples():
            access_path2val = row._asdict()
            access_path2val.pop('index', None)
            doc_dict = _access_path_dict_to_nested_dict(access_path2val)
            docs.append(doc_type.parse_obj(doc_dict))

        return docs

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Save a DocList to a `pandas.DataFrame`.
        The field names will be stored as column names. Each row of the dataframe corresponds
        to the information of one Document.
        Columns for nested fields will be named after the "__"-seperated access paths,
        such as `'image__url'` for `image.url`.

        :return: `pandas.DataFrame`
        """
        if TYPE_CHECKING:
            import pandas as pd
        else:
            pd = import_library('pandas', raise_error=True)

        if self.doc_type == AnyDoc:
            raise TypeError(
                'DocList must be homogeneous to be converted to a DataFrame.'
                'There is no document schema defined. '
                'Please specify the DocList\'s Document type using `DocList[MyDoc]`.'
            )

        fields = self.doc_type._get_access_paths()
        df = pd.DataFrame(columns=fields)

        for doc in self:
            doc_dict = _dict_to_access_paths(doc.dict())
            doc_dict = {k: [v] for k, v in doc_dict.items()}
            df = pd.concat([df, pd.DataFrame.from_dict(doc_dict)], ignore_index=True)

        return df

    # Methods to load from/to files in different formats
    @property
    def _stream_header(self) -> bytes:
        # Binary format for streaming case

        # V2 DocList streaming serialization format
        # | 1 byte | 8 bytes | 4 bytes | variable(DocArray >=0.30) | 4 bytes | variable(DocArray >=0.30) ...

        # 1 byte (uint8)
        version_byte = b'\x02'
        # 8 bytes (uint64)
        num_docs_as_bytes = len(self).to_bytes(8, 'big', signed=False)
        return version_byte + num_docs_as_bytes

    @classmethod
    @abstractmethod
    def _get_proto_class(cls: Type[T]):
        ...

    @classmethod
    def _load_binary_all(
        cls: Type[T],
        file_ctx: Union[ContextManager[io.BufferedReader], ContextManager[bytes]],
        protocol: Optional[str],
        compress: Optional[str],
        show_progress: bool,
        tensor_type: Optional[Type['AbstractTensor']] = None,
    ):
        """Read a `DocList` object from a binary file
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param tensor_type: only relevant for DocVec; tensor_type of the DocVec
        :return: a `DocList`
        """
        with file_ctx as fp:
            if isinstance(fp, bytes):
                d = fp
            else:
                d = fp.read()

        if protocol is not None and protocol in (
            'pickle-array',
            'protobuf-array',
            'json-array',
        ):
            if _get_compress_ctx(algorithm=compress) is not None:
                d = _decompress_bytes(d, algorithm=compress)
                compress = None

        if protocol is not None and protocol == 'protobuf-array':
            proto = cls._get_proto_class()()
            proto.ParseFromString(d)

            if tensor_type is not None:
                cls_ = cast('IOMixinDocVec', cls)
                return cls_.from_protobuf(proto, tensor_type=tensor_type)
            else:
                return cls.from_protobuf(proto)
        elif protocol is not None and protocol == 'pickle-array':
            return pickle.loads(d)

        elif protocol is not None and protocol == 'json-array':
            if tensor_type is not None:
                cls_ = cast('IOMixinDocVec', cls)
                return cls_.from_json(d, tensor_type=tensor_type)
            else:
                return cls.from_json(d)

        # Binary format for streaming case
        else:
            from rich import filesize

            from docarray.utils._internal.progress_bar import _get_progressbar

            # 1 byte (uint8)
            version_num = int.from_bytes(d[0:1], 'big', signed=False)
            if version_num != 2:
                raise ValueError(
                    f'Unsupported version number {version_num} in binary format, expected 2'
                )

            # 8 bytes (uint64)
            num_docs = int.from_bytes(d[1:9], 'big', signed=False)

            pbar, t = _get_progressbar(
                'Deserializing', disable=not show_progress, total=num_docs
            )

            # this 9 is version + num_docs bytes used
            start_pos = 9
            docs = []
            with pbar:
                _total_size = 0
                pbar.start_task(t)

                for _ in range(num_docs):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        d[start_pos : start_pos + 4], 'big', signed=False
                    )
                    start_doc_pos = start_pos + 4
                    end_doc_pos = start_doc_pos + len_current_doc_in_bytes
                    start_pos = end_doc_pos

                    # variable length bytes doc
                    load_protocol: str = protocol or 'protobuf'
                    doc = cls.doc_type.from_bytes(
                        d[start_doc_pos:end_doc_pos],
                        protocol=load_protocol,
                        compress=compress,
                    )
                    docs.append(doc)
                    _total_size += len_current_doc_in_bytes
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )
            if tensor_type is not None:
                cls__ = cast(Type['DocVec'], cls)
                # mypy doesn't realize that cls_ is callable
                return cls__(docs, tensor_type=tensor_type)  # type: ignore
            return cls(docs)

    @classmethod
    def _load_binary_stream(
        cls: Type[T],
        file_ctx: ContextManager[io.BufferedReader],
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> Generator['T_doc', None, None]:
        """Yield `Document` objects from a binary file

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: a generator of `Document` objects
        """

        from rich import filesize

        with file_ctx as f:
            version_numdocs_lendoc0 = f.read(9)
            # 1 byte (uint8)
            version_num = int.from_bytes(
                version_numdocs_lendoc0[0:1], 'big', signed=False
            )
            if version_num != 2:
                raise ValueError(
                    f'Unsupported version number {version_num} in binary format, expected 2'
                )

            # 8 bytes (uint64)
            num_docs = int.from_bytes(version_numdocs_lendoc0[1:9], 'big', signed=False)

            if show_progress:
                from docarray.utils._internal.progress_bar import _get_progressbar

                pbar, t = _get_progressbar(
                    'Deserializing', disable=not show_progress, total=num_docs
                )
            else:
                from contextlib import nullcontext

                pbar = nullcontext()

            with pbar:
                if show_progress:
                    _total_size = 0
                    pbar.start_task(t)
                for _ in range(num_docs):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        f.read(4), 'big', signed=False
                    )
                    load_protocol: str = protocol
                    yield cls.doc_type.from_bytes(
                        f.read(len_current_doc_in_bytes),
                        protocol=load_protocol,
                        compress=compress,
                    )
                    if show_progress:
                        _total_size += len_current_doc_in_bytes
                        pbar.update(
                            t, advance=1, total_size=str(filesize.decimal(_total_size))
                        )

    @staticmethod
    def _get_file_context(
        file: Union[str, bytes, pathlib.Path, io.BufferedReader, _LazyRequestReader],
        protocol: str,
        compress: Optional[str] = None,
    ) -> Tuple[Union[nullcontext, io.BufferedReader], Optional[str], Optional[str]]:
        load_protocol: Optional[str] = protocol
        load_compress: Optional[str] = compress
        file_ctx: Union[nullcontext, io.BufferedReader]
        if isinstance(file, (io.BufferedReader, _LazyRequestReader, bytes)):
            file_ctx = nullcontext(file)
        # by checking path existence we allow file to be of type Path, LocalPath, PurePath and str
        elif isinstance(file, (str, pathlib.Path)) and os.path.exists(file):
            load_protocol, load_compress = _protocol_and_compress_from_file_path(
                file, protocol, compress
            )
            file_ctx = open(file, 'rb')
        else:
            raise FileNotFoundError(f'cannot find file {file}')
        return file_ctx, load_protocol, load_compress

    @classmethod
    def load_binary(
        cls: Type[T],
        file: Union[str, bytes, pathlib.Path, io.BufferedReader, _LazyRequestReader],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
        streaming: bool = False,
    ) -> Union[T, Generator['T_doc', None, None]]:
        """Load doc_list elements from a compressed binary file.

        In case protocol is pickle the `Documents` are streamed from disk to save memory usage

        !!! note
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be loaded assuming `protocol=protobuf`
            and `compress=lz4`.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.

        :return: a `DocList` object

        """
        file_ctx, load_protocol, load_compress = cls._get_file_context(
            file, protocol, compress
        )
        if streaming:
            if load_protocol not in SINGLE_PROTOCOLS:
                raise ValueError(
                    f'`streaming` is only available when using {" or ".join(map(lambda x: f"`{x}`", SINGLE_PROTOCOLS))} as protocol, '
                    f'got {load_protocol}'
                )
            else:
                return cls._load_binary_stream(
                    file_ctx,
                    protocol=load_protocol,
                    compress=load_compress,
                    show_progress=show_progress,
                )
        else:
            return cls._load_binary_all(
                file_ctx, load_protocol, load_compress, show_progress
            )

    def save_binary(
        self,
        file: Union[str, pathlib.Path],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> None:
        """Save DocList into a binary file.

        It will use the protocol to pick how to save the DocList.
        If used `picke-doc_list` and `protobuf-array` the DocList will be stored
        and compressed at complete level using `pickle` or `protobuf`.
        When using `protobuf` or `pickle` as protocol each Document in DocList
        will be stored individually and this would make it available for streaming.

         !!! note
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be created using `protocol=protobuf`
            and `compress=lz4`.

        :param file: File or filename to which the data is saved.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        """
        if isinstance(file, io.BufferedWriter):
            file_ctx = nullcontext(file)
        else:
            _protocol, _compress = _protocol_and_compress_from_file_path(file)

            if _protocol is not None:
                protocol = _protocol
            if _compress is not None:
                compress = _compress

            file_ctx = open(file, 'wb')

        self.to_bytes(
            protocol=protocol,
            compress=compress,
            file_ctx=file_ctx,
            show_progress=show_progress,
        )
