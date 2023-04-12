import csv
import io
import pathlib
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_inspect import is_union_type

from docarray.array.any_array import AnyDocArray
from docarray.array.doc_list.io import IOMixinArray, _LazyRequestReader
from docarray.array.doc_list.pushpull import PushPullMixin
from docarray.array.doc_list.sequence_indexing_mixin import (
    IndexingSequenceMixin,
    IndexIterType,
)
from docarray.base_doc import AnyDoc, BaseDoc
from docarray.typing import NdArray

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.array.doc_vec.doc_vec import DocVec
    from docarray.proto import DocListProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='DocList')
T_doc = TypeVar('T_doc', bound=BaseDoc)


def _delegate_meth_to_data(meth_name: str) -> Callable:
    """
    create a function that mimic a function call to the data attribute of the
    DocList

    :param meth_name: name of the method
    :return: a method that mimic the meth_name
    """
    func = getattr(list, meth_name)

    @wraps(func)
    def _delegate_meth(self, *args, **kwargs):
        return getattr(self._data, meth_name)(*args, **kwargs)

    return _delegate_meth


class DocList(
    IndexingSequenceMixin[T_doc], PushPullMixin, IOMixinArray, AnyDocArray[T_doc]
):
    """
     DocList is a container of Documents.

    A DocList is a list of Documents of any schema. However, many
    DocList features are only available if these Documents are
    homogeneous and follow the same schema. To precise this schema you can use
    the `DocList[MyDocument]` syntax where MyDocument is a Document class
    (i.e. schema). This creates a DocList that can only contains Documents of
    the type `MyDocument`.


    ```python
    from docarray import BaseDoc, DocList
    from docarray.typing import NdArray, ImageUrl
    from typing import Optional


    class Image(BaseDoc):
        tensor: Optional[NdArray[100]]
        url: ImageUrl


    docs = DocList[Image](
        Image(url='http://url.com/foo.png') for _ in range(10)
    )  # noqa: E510


    # If your DocList is homogeneous (i.e. follows the same schema), you can access
    # fields at the DocList level (for example `docs.tensor` or `docs.url`).

    print(docs.url)
    # [ImageUrl('http://url.com/foo.png', host_type='domain'), ...]


    # You can also set fields, with `docs.tensor = np.random.random([10, 100])`:


    import numpy as np

    docs.tensor = np.random.random([10, 100])

    print(docs.tensor)
    # [NdArray([0.11299577, 0.47206767, 0.481723  , 0.34754724, 0.15016037,
    #          0.88861321, 0.88317666, 0.93845579, 0.60486676, ... ]), ...]


    # You can index into a DocList like a numpy doc_list or torch tensor:

    docs[0]  # index by position
    docs[0:5:2]  # index by slice
    docs[[0, 2, 3]]  # index by list of indices
    docs[True, False, True, True, ...]  # index by boolean mask


    # You can delete items from a DocList like a Python List

    del docs[0]  # remove first element from DocList
    del docs[0:5]  # remove elements for 0 to 5 from DocList
    ```

    :param docs: iterable of Document

    """

    doc_type: Type[BaseDoc] = AnyDoc

    def __init__(
        self,
        docs: Optional[Iterable[T_doc]] = None,
    ):
        self._data: List[T_doc] = list(self._validate_docs(docs)) if docs else []

    @classmethod
    def construct(
        cls: Type[T],
        docs: Sequence[T_doc],
    ) -> T:
        """
        Create a `DocList` without validation any data. The data must come from a
        trusted source
        :param docs: a Sequence (list) of Document with the same schema
        :return: a `DocList` object
        """
        new_docs = cls.__new__(cls)
        new_docs._data = docs if isinstance(docs, list) else list(docs)
        return new_docs

    def __eq__(self, other: Any) -> bool:
        if self.__len__() != other.__len__():
            return False
        for doc_self, doc_other in zip(self, other):
            if doc_self != doc_other:
                return False
        return True

    def _validate_docs(self, docs: Iterable[T_doc]) -> Iterable[T_doc]:
        """
        Validate if an Iterable of Document are compatible with this `DocList`
        """
        for doc in docs:
            yield self._validate_one_doc(doc)

    def _validate_one_doc(self, doc: T_doc) -> T_doc:
        """Validate if a Document is compatible with this `DocList`"""
        if not issubclass(self.doc_type, AnyDoc) and not isinstance(doc, self.doc_type):
            raise ValueError(f'{doc} is not a {self.doc_type}')
        return doc

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __bytes__(self) -> bytes:
        with io.BytesIO() as bf:
            self._write_bytes(bf=bf)
            return bf.getvalue()

    def append(self, doc: T_doc):
        """
        Append a Document to the `DocList`. The Document must be from the same class
        as the `.doc_type` of this `DocList` otherwise it will fail.
        :param doc: A Document
        """
        self._data.append(self._validate_one_doc(doc))

    def extend(self, docs: Iterable[T_doc]):
        """
        Extend a `DocList` with an Iterable of Document. The Documents must be from
        the same class as the `.doc_type` of this `DocList` otherwise it will
        fail.
        :param docs: Iterable of Documents
        """
        self._data.extend(self._validate_docs(docs))

    def insert(self, i: int, doc: T_doc):
        """
        Insert a Document to the `DocList`. The Document must be from the same
        class as the doc_type of this `DocList` otherwise it will fail.
        :param i: index to insert
        :param doc: A Document
        """
        self._data.insert(i, self._validate_one_doc(doc))

    pop = _delegate_meth_to_data('pop')
    remove = _delegate_meth_to_data('remove')
    reverse = _delegate_meth_to_data('reverse')
    sort = _delegate_meth_to_data('sort')

    def _get_data_column(
        self: T,
        field: str,
    ) -> Union[MutableSequence, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this doc_list contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the doc_list like container
        """
        field_type = self.__class__.doc_type._get_field_type(field)

        if (
            not is_union_type(field_type)
            and isinstance(field_type, type)
            and issubclass(field_type, BaseDoc)
        ):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return DocList.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self),
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_data_column(
        self: T,
        field: str,
        values: Union[List, T, 'AbstractTensor'],
    ):
        """Set all Documents in this `DocList` using the passed values

        :param field: name of the fields to set
        :values: the values to set at the `DocList` level
        """
        ...

        for doc, value in zip(self, values):
            setattr(doc, field, value)

    def stack(
        self,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> 'DocVec':
        """
        Convert the `DocList` into a `DocVec`. `Self` cannot be used
        afterwards
        :param tensor_type: Tensor Class used to wrap the doc_vec tensors. This is useful
        if the BaseDoc has some undefined tensor type like AnyTensor or Union of NdArray and TorchTensor
        :return: A `DocVec` of the same document type as self
        """
        from docarray.array.doc_vec.doc_vec import DocVec

        return DocVec.__class_getitem__(self.doc_type)(self, tensor_type=tensor_type)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[BaseDoc]],
        field: 'ModelField',
        config: 'BaseConfig',
    ):
        from docarray.array.doc_vec.doc_vec import DocVec

        if isinstance(value, (cls, DocVec)):
            return value
        elif isinstance(value, Iterable):
            return cls(value)
        else:
            raise TypeError(f'Expecting an Iterable of {cls.doc_type}')

    def traverse_flat(
        self: 'DocList',
        access_path: str,
    ) -> List[Any]:
        nodes = list(AnyDocArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocArray._flatten_one_level(nodes)

        return flattened

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocListProto') -> T:
        """create a Document from a protobuf message
        :param pb_msg: The protobuf message from where to construct the `DocList`
        """
        return super().from_protobuf(pb_msg)

    @overload
    def __getitem__(self, item: int) -> T_doc:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item):
        return super().__getitem__(item)

    ########################################################################################################################################################
    ### this section is just for documentation purposes will be removed later once https://github.com/mkdocstrings/griffe/issues/138 is fixed ##############
    ########################################################################################################################################################

    def to_protobuf(self) -> 'DocListProto':
        """Convert DocList into a Protobuf message"""
        return super(DocList, self).to_protobuf()

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> T:
        """Deserialize bytes into a DocList.

        :param data: Bytes from which to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized DocList
        """
        return super(DocList, cls).from_bytes(
            data, protocol=protocol, compress=compress, show_progress=show_progress
        )

    def to_binary_stream(
        self,
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> Iterator[bytes]:
        return super().to_binary_stream(
            protocol=protocol, compress=compress, show_progress=show_progress
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
        :param compress: compress algorithm to use between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param file_ctx: File or filename or serialized bytes where the data is stored.
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the binary serialization in bytes or None if file_ctx is passed where to store
        """
        return super().to_bytes(
            protocol=protocol,
            compress=compress,
            file_ctx=file_ctx,
            show_progress=show_progress,
        )

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
    ) -> T:
        """Deserialize base64 strings into a DocList.

        :param data: Base64 string to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :return: the deserialized DocList
        """
        return super(DocList, cls).from_base64(
            data, protocol=protocol, compress=compress, show_progress=show_progress
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
        return super().to_base64(
            protocol=protocol, compress=compress, show_progress=show_progress
        )

    @classmethod
    def from_json(
        cls: Type[T],
        file: Union[str, bytes, bytearray],
    ) -> T:
        """Deserialize JSON strings or bytes into a DocList.

        :param file: JSON object from where to deserialize a DocList
        :return: the deserialized DocList
        """
        return super(DocList, cls).from_json(file)

    def to_json(self) -> bytes:
        """Convert the object into JSON bytes. Can be loaded via :meth:`.from_json`.
        :return: JSON serialization of DocList
        """
        return super().to_json()

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        encoding: str = 'utf-8',
        dialect: Union[str, csv.Dialect] = 'excel',
    ) -> 'DocList':
        """
        Load a DocList from a csv file following the schema defined in the
        :attr:`~docarray.DocList.doc_type` attribute.
        Every row of the csv file will be mapped to one document in the doc_list.
        The column names (defined in the first row) have to match the field names
        of the Document type.
        For nested fields use "__"-separated access paths, such as 'image__url'.

        List-like fields (including field of type DocList) are not supported.

        :param file_path: path to csv file to load DocList from.
        :param encoding: encoding used to read the csv file. Defaults to 'utf-8'.
        :param dialect: defines separator and how to handle whitespaces etc.
            Can be a csv.Dialect instance or one string of:
            'excel' (for comma seperated values),
            'excel-tab' (for tab separated values),
            'unix' (for csv file generated on UNIX systems).
        :return: DocList
        """
        return super(DocList, cls).from_csv(
            file_path, encoding=encoding, dialect=dialect
        )

    def to_csv(
        self, file_path: str, dialect: Union[str, csv.Dialect] = 'excel'
    ) -> None:
        """
        Save a DocList to a csv file.
        The field names will be stored in the first row. Each row corresponds to the
        information of one Document.
        Columns for nested fields will be named after the "__"-seperated access paths,
        such as `"image__url"` for `image.url`.

        :param file_path: path to a csv file.
        :param dialect: defines separator and how to handle whitespaces etc.
            Can be a csv.Dialect instance or one string of:
            'excel' (for comma seperated values),
            'excel-tab' (for tab separated values),
            'unix' (for csv file generated on UNIX systems).
        """
        return super().to_csv(file_path, dialect=dialect)

    @classmethod
    def from_dataframe(cls, df: 'pd.DataFrame') -> 'DocList':
        """
        Load a DocList from a `pandas.DataFrame` following the schema
        defined in the :attr:`~docarray.DocList.doc_type` attribute.
        Every row of the dataframe will be mapped to one Document in the doc_list.
        The column names of the dataframe have to match the field names of the
        Document type.
        For nested fields use "__"-separated access paths as column names,
        such as 'image__url'.

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

        :param df: pandas.DataFrame to extract Document's information from
        :return: DocList where each Document contains the information of one
            corresponding row of the `pandas.DataFrame`.
        """
        return super(DocList, cls).from_dataframe(df)

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Save a DocList to a `pandas.DataFrame`.
        The field names will be stored as column names. Each row of the dataframe corresponds
        to the information of one Document.
        Columns for nested fields will be named after the "__"-seperated access paths,
        such as `"image__url"` for `image.url`.

        :return: pandas.DataFrame
        """
        return super().to_dataframe()

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

        :param file: File or filename or serialized bytes where the data is stored.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between 'lz4', 'gzip', 'bz2', 'zstd', 'lzma'
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param streaming: if `True` returns a generator over `Document` objects.
        In case protocol is pickle the `Documents` are streamed from disk to save memory usage
        :return: a DocList object

        .. note::
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be loaded assuming `protocol=protobuf`
            and `compress=lz4`.
        """
        return super().load_binary(
            file, protocol=protocol, compress=compress, show_progress=show_progress
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
        If used 'picke-doc_list` and `protobuf-array` the DocList will be stored
        and compressed at complete level using `pickle` or `protobuf`.
        When using `protobuf` or `pickle` as protocol each Document in DocList
        will be stored individually and this would make it available for streaming.

        !! note
            If `file` is `str` it can specify `protocol` and `compress` as file extensions.
            This functionality assumes `file=file_name.$protocol.$compress` where `$protocol` and `$compress` refer to a
            string interpolation of the respective `protocol` and `compress` methods.
            For example if `file=my_docarray.protobuf.lz4` then the binary data will be created using `protocol=protobuf`
            and `compress=lz4`.

        :param file: File or filename to which the data is saved.
        :param protocol: protocol to use. It can be 'pickle-array', 'protobuf-array', 'pickle' or 'protobuf'
        :param compress: compress algorithm to use between : `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`


        """
        return super().save_binary(
            file, protocol=protocol, compress=compress, show_progress=show_progress
        )
