import base64
import io
import json
import os
import pathlib
import pickle
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    ContextManager,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from typing_inspect import is_union_type

from docarray.array.abstract_array import AnyDocumentArray
from docarray.base_document import AnyDocument, BaseDocument
from docarray.typing import NdArray
from docarray.utils.compress import _decompress_bytes, _get_compress_ctx
from docarray.utils.misc import is_torch_available

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.array.stacked.array_stacked import DocumentArrayStacked
    from docarray.proto import DocumentArrayProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='DocumentArray')
T_doc = TypeVar('T_doc', bound=BaseDocument)
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]

ARRAY_PROTOCOLS = {'protobuf-array', 'pickle-array'}
SINGLE_PROTOCOLS = {'pickle', 'protobuf'}
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


def _delegate_meth_to_data(meth_name: str) -> Callable:
    """
    create a function that mimic a function call to the data attribute of the
    DocumentArray

    :param meth_name: name of the method
    :return: a method that mimic the meth_name
    """
    func = getattr(list, meth_name)

    @wraps(func)
    def _delegate_meth(self, *args, **kwargs):
        return getattr(self._data, meth_name)(*args, **kwargs)

    return _delegate_meth


def _is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it


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


class DocumentArray(AnyDocumentArray, Generic[T_doc]):
    """
     DocumentArray is a container of Documents.

    :param docs: iterable of Document
    :param tensor_type: Class used to wrap the tensors of the Documents when stacked

    A DocumentArray is a list of Documents of any schema. However, many
    DocumentArray features are only available if these Documents are
    homogeneous and follow the same schema. To precise this schema you can use
    the `DocumentArray[MyDocument]` syntax where MyDocument is a Document class
    (i.e. schema). This creates a DocumentArray that can only contains Documents of
    the type 'MyDocument'.

    EXAMPLE USAGE
    .. code-block:: python
        from docarray import BaseDocument, DocumentArray
        from docarray.typing import NdArray, ImageUrl
        from typing import Optional


        class Image(BaseDocument):
            tensor: Optional[NdArray[100]]
            url: ImageUrl


        da = DocumentArray[Image](
            Image(url='http://url.com/foo.png') for _ in range(10)
        )  # noqa: E510


    If your DocumentArray is homogeneous (i.e. follows the same schema), you can access
    fields at the DocumentArray level (for example `da.tensor` or `da.url`).
    You can also set fields, with `da.tensor = np.random.random([10, 100])`:


    .. code-block:: python
        print(da.url)
        # [ImageUrl('http://url.com/foo.png', host_type='domain'), ...]
        import numpy as np

        da.tensor = np.random.random([10, 100])
        print(da.tensor)
        # [NdArray([0.11299577, 0.47206767, 0.481723  , 0.34754724, 0.15016037,
        #          0.88861321, 0.88317666, 0.93845579, 0.60486676, ... ]), ...]

    You can index into a DocumentArray like a numpy array or torch tensor:

    .. code-block:: python
        da[0]  # index by position
        da[0:5:2]  # index by slice
        da[[0, 2, 3]]  # index by list of indices
        da[True, False, True, True, ...]  # index by boolean mask

    You can delete items from a DocumentArray like a Python List

    .. code-block:: python
        del da[0]  # remove first element from DocumentArray
        del da[0:5]  # remove elements fro 0 to 5 from DocumentArray

    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(
        self,
        docs: Optional[Iterable[T_doc]] = None,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):

        self._data: List[T_doc] = list(self._validate_docs(docs)) if docs else []
        self.tensor_type = tensor_type

    def _validate_docs(self, docs: Iterable[T_doc]) -> Iterable[T_doc]:
        """
        Validate if an Iterable of Document are compatible with this DocumentArray
        """
        for doc in docs:
            yield self._validate_one_doc(doc)

    def _validate_one_doc(self, doc: T_doc) -> T_doc:
        """Validate if a Document is compatible with this DocumentArray"""
        if not issubclass(self.document_type, AnyDocument) and not isinstance(
            doc, self.document_type
        ):
            raise ValueError(f'{doc} is not a {self.document_type}')
        return doc

    def __len__(self):
        return len(self._data)

    @overload
    def __getitem__(self: T, item: int) -> BaseDocument:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item):
        item = self._normalize_index_item(item)

        if type(item) == slice:
            return self.__class__(self._data[item])

        if isinstance(item, int):
            return self._data[item]

        if item is None:
            return self

        # _normalize_index_item() guarantees the line below is correct
        head = item[0]  # type: ignore
        if isinstance(head, bool):
            return self._get_from_mask(item)
        elif isinstance(head, int):
            return self._get_from_indices(item)
        else:
            raise TypeError(f'Invalid type {type(head)} for indexing')

    def __setitem__(self: T, key: IndexIterType, value: Union[T, BaseDocument]):
        key_norm = self._normalize_index_item(key)

        if isinstance(key_norm, int):
            value_int = cast(BaseDocument, value)
            self._data[key_norm] = value_int
        elif isinstance(key_norm, slice):
            value_slice = cast(T, value)
            self._data[key_norm] = value_slice
        else:
            # _normalize_index_item() guarantees the line below is correct
            head = key_norm[0]  # type: ignore
            if isinstance(head, bool):
                key_norm_ = cast(Iterable[bool], key_norm)
                value_ = cast(Sequence[BaseDocument], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_mask(key_norm_, value_)
            elif isinstance(head, int):
                key_norm__ = cast(Iterable[int], key_norm)
                value_ = cast(Sequence[BaseDocument], value)  # this is no strictly true
                # set_by_mask requires value_ to have getitem which
                # _normalize_index_item() ensures
                return self._set_by_indices(key_norm__, value_)
            else:
                raise TypeError(f'Invalid type {type(head)} for indexing')

    def __iter__(self):
        return iter(self._data)

    @overload
    def __delitem__(self: T, key: int) -> None:
        ...

    @overload
    def __delitem__(self: T, key: IndexIterType) -> None:
        ...

    def __delitem__(self, key) -> None:
        key = self._normalize_index_item(key)

        if key is None:
            return

        del self._data[key]

    def __bytes__(self) -> bytes:
        with io.BytesIO() as bf:
            self._write_bytes(bf=bf)
            return bf.getvalue()

    @staticmethod
    def _normalize_index_item(
        item: Any,
    ) -> Union[int, slice, Iterable[int], Iterable[bool], None]:
        # basic index types
        if item is None or isinstance(item, (int, slice, tuple, list)):
            return item

        # numpy index types
        if _is_np_int(item):
            return item.item()

        index_has_getitem = hasattr(item, '__getitem__')
        is_valid_bulk_index = index_has_getitem and isinstance(item, Iterable)
        if not is_valid_bulk_index:
            raise ValueError(f'Invalid index type {type(item)}')

        if isinstance(item, np.ndarray) and (
            item.dtype == np.bool_ or np.issubdtype(item.dtype, np.integer)
        ):
            return item.tolist()

        # torch index types
        torch_available = is_torch_available()
        if torch_available:
            import torch
        else:
            raise ValueError(f'Invalid index type {type(item)}')
        allowed_torch_dtypes = [
            torch.bool,
            torch.int64,
        ]
        if isinstance(item, torch.Tensor) and (item.dtype in allowed_torch_dtypes):
            return item.tolist()

        return item

    def _get_from_indices(self: T, item: Iterable[int]) -> T:
        results = []
        for ix in item:
            results.append(self._data[ix])
        return self.__class__(results)

    def _set_by_indices(self: T, item: Iterable[int], value: Iterable[BaseDocument]):
        # here we cannot use _get_offset_to_doc() because we need to change the doc
        # that a given offset points to, not just retrieve it.
        # Future optimization idea: _data could be List[DocContainer], where
        # DocContainer points to the doc. Then we could use _get_offset_to_container()
        # to swap the doc in the container.
        for ix, doc_to_set in zip(item, value):
            try:
                self._data[ix] = doc_to_set
            except KeyError:
                raise IndexError(f'Index {ix} is out of range')

    def _get_from_mask(self: T, item: Iterable[bool]) -> T:
        return self.__class__(
            (doc for doc, mask_value in zip(self, item) if mask_value)
        )

    def _set_by_mask(self: T, item: Iterable[bool], value: Sequence[BaseDocument]):
        i_value = 0
        for i, mask_value in zip(range(len(self)), item):
            if mask_value:
                self._data[i] = value[i_value]
                i_value += 1

    def append(self, doc: T_doc):
        """
        Append a Document to the DocumentArray. The Document must be from the same class
        as the document_type of this DocumentArray otherwise it will fail.
        :param doc: A Document
        """
        self._data.append(self._validate_one_doc(doc))

    def extend(self, docs: Iterable[T_doc]):
        """
        Extend a DocumentArray with an Iterable of Document. The Documents must be from
        the same class as the document_type of this DocumentArray otherwise it will
        fail.
        :param docs: Iterable of Documents
        """
        self._data.extend(self._validate_docs(docs))

    def insert(self, i: int, doc: T_doc):
        """
        Insert a Document to the DocumentArray. The Document must be from the same
        class as the document_type of this DocumentArray otherwise it will fail.
        :param i: index to insert
        :param doc: A Document
        """
        self._data.insert(i, self._validate_one_doc(doc))

    pop = _delegate_meth_to_data('pop')
    remove = _delegate_meth_to_data('remove')
    reverse = _delegate_meth_to_data('reverse')
    sort = _delegate_meth_to_data('sort')

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        field_type = self.__class__.document_type._get_field_type(field)

        if (
            not is_union_type(field_type)
            and isinstance(field_type, type)
            and issubclass(field_type, BaseDocument)
        ):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return DocumentArray.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self), tensor_type=self.tensor_type
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, 'AbstractTensor'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        ...

        for doc, value in zip(self, values):
            setattr(doc, field, value)

    @contextmanager
    def stacked_mode(self):
        """
        Context manager to convert DocumentArray to a DocumentArrayStacked and unstack
        it when exiting the context manager.
        EXAMPLE USAGE
        .. code-block:: python
            with da.stacked_mode():
                ...
        """

        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        try:
            da_stacked = DocumentArrayStacked.__class_getitem__(self.document_type)(
                self,
            )
            yield da_stacked
        finally:
            self = DocumentArrayStacked.__class_getitem__(self.document_type).unstack(
                da_stacked
            )

    def stack(self) -> 'DocumentArrayStacked':
        """
        Convert the DocumentArray into a DocumentArrayStacked. `Self` cannot be used
        afterwards
        """
        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        return DocumentArrayStacked.__class_getitem__(self.document_type)(self)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[BaseDocument]],
        field: 'ModelField',
        config: 'BaseConfig',
    ):
        from docarray.array.stacked.array_stacked import DocumentArrayStacked

        if isinstance(value, (cls, DocumentArrayStacked)):
            return value
        elif isinstance(value, Iterable):
            return cls(value)
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def traverse_flat(
        self: 'DocumentArray',
        access_path: str,
    ) -> List[Any]:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        return flattened

    # Methods to load from/to different formats

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message
        :param pb_msg: The protobuf message from where to construct the DocumentArray
        """
        return cls(
            cls.document_type.from_protobuf(doc_proto) for doc_proto in pb_msg.docs
        )

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import DocumentArrayProto

        da_proto = DocumentArrayProto()
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
        """Deserialize bytes into a DocumentArray.

        :param data: Bytes from which to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized DocumentArray
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
            elif protocol in SINGLE_PROTOCOLS:
                from rich import filesize

                from docarray.utils.progress_bar import _get_progressbar

                pbar, t = _get_progressbar(
                    'Serializing', disable=not show_progress, total=len(self)
                )

                f.write(self._stream_header)

                with pbar:
                    _total_size = 0
                    pbar.start_task(t)
                    for doc in self:
                        doc_bytes = doc.to_bytes(protocol=protocol, compress=compress)
                        len_doc_as_bytes = len(doc_bytes).to_bytes(
                            4, 'big', signed=False
                        )
                        all_bytes = len_doc_as_bytes + doc_bytes
                        f.write(all_bytes)
                        _total_size += len(all_bytes)
                        pbar.update(
                            t,
                            advance=1,
                            total_size=str(filesize.decimal(_total_size)),
                        )
            else:
                raise ValueError(
                    f'protocol={protocol} is not supported. Can be only {ALLOWED_PROTOCOLS}.'
                )

    def to_bytes(
        self,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        file_ctx: Optional[BinaryIO] = None,
        show_progress: bool = False,
    ) -> Optional[bytes]:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :param file_ctx: File or filename or serialized bytes where the data is stored.
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the binary serialization in bytes or None if file_ctx is passed where to store
        """

        with (file_ctx or io.BytesIO()) as bf:
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
        """Deserialize base64 strings into a DocumentArray.

        :param data: Base64 string to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized DocumentArray
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
        :param compress: compress algorithm to use
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
        """Deserialize JSON strings or bytes into a DocumentArray.

        :param file: JSON object from where to deserialize a DocumentArray
        :return: the deserialized DocumentArray
        """
        json_docs = json.loads(file)
        return cls([cls.document_type.parse_raw(v) for v in json_docs])

    def to_json(self) -> str:
        """Convert the object into a JSON string. Can be loaded via :meth:`.from_json`.
        :return: JSON serialization of DocumentArray
        """
        return json.dumps([doc.json() for doc in self])

    # Methods to load from/to files in different formats
    @property
    def _stream_header(self) -> bytes:
        # Binary format for streaming case

        # V1 DocArray streaming serialization format
        # | 1 byte | 8 bytes | 4 bytes | variable | 4 bytes | variable ...

        # 1 byte (uint8)
        version_byte = b'\x01'
        # 8 bytes (uint64)
        num_docs_as_bytes = len(self).to_bytes(8, 'big', signed=False)
        return version_byte + num_docs_as_bytes

    @classmethod
    def _load_binary_all(
        cls: Type[T],
        file_ctx: Union[ContextManager[io.BufferedReader], ContextManager[bytes]],
        protocol: Optional[str],
        compress: Optional[str],
        show_progress: bool,
    ):
        """Read a `DocumentArray` object from a binary file
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: a `DocumentArray`
        """
        with file_ctx as fp:
            if isinstance(fp, bytes):
                d = fp
            else:
                d = fp.read()

        if protocol is not None and protocol in ('pickle-array', 'protobuf-array'):
            if _get_compress_ctx(algorithm=compress) is not None:
                d = _decompress_bytes(d, algorithm=compress)
                compress = None

        if protocol is not None and protocol == 'protobuf-array':
            from docarray.proto import DocumentArrayProto

            dap = DocumentArrayProto()
            dap.ParseFromString(d)

            return cls.from_protobuf(dap)
        elif protocol is not None and protocol == 'pickle-array':
            return pickle.loads(d)

        # Binary format for streaming case
        else:
            from rich import filesize

            from docarray.utils.progress_bar import _get_progressbar

            # 1 byte (uint8)
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
                    doc = cls.document_type.from_bytes(
                        d[start_doc_pos:end_doc_pos],
                        protocol=load_protocol,
                        compress=compress,
                    )
                    docs.append(doc)
                    _total_size += len_current_doc_in_bytes
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )
            return cls(docs)

    @classmethod
    def _load_binary_stream(
        cls: Type[T],
        file_ctx: ContextManager[io.BufferedReader],
        protocol: Optional[str] = None,
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> Generator['BaseDocument', None, None]:
        """Yield `Document` objects from a binary file

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: a generator of `Document` objects
        """

        from rich import filesize

        from docarray import BaseDocument
        from docarray.utils.progress_bar import _get_progressbar

        with file_ctx as f:
            version_numdocs_lendoc0 = f.read(9)
            # 1 byte (uint8)
            # 8 bytes (uint64)
            num_docs = int.from_bytes(version_numdocs_lendoc0[1:9], 'big', signed=False)

            pbar, t = _get_progressbar(
                'Deserializing', disable=not show_progress, total=num_docs
            )

            with pbar:
                _total_size = 0
                pbar.start_task(t)
                for _ in range(num_docs):
                    # 4 bytes (uint32)
                    len_current_doc_in_bytes = int.from_bytes(
                        f.read(4), 'big', signed=False
                    )
                    _total_size += len_current_doc_in_bytes
                    load_protocol: str = protocol or 'protobuf'
                    yield BaseDocument.from_bytes(
                        f.read(len_current_doc_in_bytes),
                        protocol=load_protocol,
                        compress=compress,
                    )
                    pbar.update(
                        t, advance=1, total_size=str(filesize.decimal(_total_size))
                    )

    @classmethod
    def load_binary(
        cls: Type[T],
        file: Union[str, bytes, pathlib.Path, io.BufferedReader, _LazyRequestReader],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
        streaming: bool = False,
    ) -> Union[T, Generator['BaseDocument', None, None]]:
        """Load array elements from a compressed binary file.

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.
        In case protocol is pickle the `Documents` are streamed from disk to save memory usage
        :return: a DocumentArray object

        .. note::
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be loaded assuming `protocol=protobuf`
            and `compress=lz4`.
        """
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
        if streaming:
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
        """Save DocumentArray into a binary file.

        It will use the protocol to pick how to save the DocumentArray.
        If used 'picke-array` and `protobuf-array` the DocumentArray will be stored
        and compressed at complete level using `pickle` or `protobuf`.
        When using `protobuf` or `pickle` as protocol each Document in DocumentArray
        will be stored individually and this would make it available for streaming.

        :param file: File or filename to which the data is saved.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`

         .. note::
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be created using `protocol=protobuf`
            and `compress=lz4`.
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
