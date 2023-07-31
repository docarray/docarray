import base64
import io
import pathlib
from abc import abstractmethod
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import orjson
from pydantic import parse_obj_as

from docarray.array.doc_list.io import (
    SINGLE_PROTOCOLS,
    IOMixinDocList,
    _LazyRequestReader,
)
from docarray.array.doc_vec.column_storage import ColumnStorage
from docarray.array.list_advance_indexing import ListAdvancedIndexing
from docarray.base_doc import BaseDoc
from docarray.base_doc.mixins.io import _type_to_protobuf
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    import csv

    import pandas as pd

    from docarray.array.doc_vec.doc_vec import DocVec
    from docarray.proto import (
        DocVecProto,
        ListOfDocArrayProto,
        ListOfDocVecProto,
        NdArrayProto,
    )


T = TypeVar('T', bound='IOMixinDocVec')
T_doc = TypeVar('T_doc', bound=BaseDoc)

NONE_NDARRAY_PROTO_SHAPE = (0,)
NONE_NDARRAY_PROTO_DTYPE = 'None'


def _none_ndarray_proto() -> 'NdArrayProto':
    from docarray.proto import NdArrayProto

    zeros_arr = parse_obj_as(NdArray, np.zeros(NONE_NDARRAY_PROTO_SHAPE))
    nd_proto = NdArrayProto()
    nd_proto.dense.buffer = zeros_arr.tobytes()
    nd_proto.dense.ClearField('shape')
    nd_proto.dense.shape.extend(list(zeros_arr.shape))
    nd_proto.dense.dtype = NONE_NDARRAY_PROTO_DTYPE

    return nd_proto


def _none_docvec_proto() -> 'DocVecProto':
    from docarray.proto import DocVecProto

    return DocVecProto()


def _none_list_of_docvec_proto() -> 'ListOfDocArrayProto':
    from docarray.proto import ListOfDocVecProto

    return ListOfDocVecProto()


def _is_none_ndarray_proto(proto: 'NdArrayProto') -> bool:
    return (
        proto.dense.shape == list(NONE_NDARRAY_PROTO_SHAPE)
        and proto.dense.dtype == NONE_NDARRAY_PROTO_DTYPE
    )


def _is_none_docvec_proto(proto: 'DocVecProto') -> bool:
    return (
        proto.tensor_columns == {}
        and proto.doc_columns == {}
        and proto.docs_vec_columns == {}
        and proto.any_columns == {}
    )


def _is_none_list_of_docvec_proto(proto: 'ListOfDocVecProto') -> bool:
    from docarray.proto import ListOfDocVecProto

    return isinstance(proto, ListOfDocVecProto) and len(proto.data) == 0


class IOMixinDocVec(IOMixinDocList):
    @classmethod
    @abstractmethod
    def from_columns_storage(cls: Type[T], storage: ColumnStorage) -> T:
        ...

    @classmethod
    @abstractmethod
    def __class_getitem__(cls, item: Union[Type[BaseDoc], TypeVar, str]):
        ...

    @classmethod
    def from_json(
        cls: Type[T],
        file: Union[str, bytes, bytearray],
        tensor_type: Type[AbstractTensor] = NdArray,
    ) -> T:
        """Deserialize JSON strings or bytes into a `DocList`.

        :param file: JSON object from where to deserialize a `DocList`
        :param tensor_type: the tensor type to use for the tensor columns.
            Could be NdArray, TorchTensor, or TensorFlowTensor. Defaults to NdArray.
            All tensors of the output DocVec will be of this type.
        :return: the deserialized `DocList`
        """
        json_columns = orjson.loads(file)
        return cls._from_json_col_dict(json_columns, tensor_type=tensor_type)

    @classmethod
    def _from_json_col_dict(
        cls: Type[T],
        json_columns: Dict[str, Any],
        tensor_type: Type[AbstractTensor] = NdArray,
    ) -> T:

        tensor_cols = json_columns['tensor_columns']
        doc_cols = json_columns['doc_columns']
        docs_vec_cols = json_columns['docs_vec_columns']
        any_cols = json_columns['any_columns']

        for key, col in tensor_cols.items():
            if col is not None:
                tensor_cols[key] = parse_obj_as(tensor_type, col)
            else:
                tensor_cols[key] = None

        for key, col in doc_cols.items():
            if col is not None:
                col_doc_type = cls.doc_type._get_field_type(key)
                doc_cols[key] = cls.__class_getitem__(col_doc_type)._from_json_col_dict(
                    col, tensor_type=tensor_type
                )
            else:
                doc_cols[key] = None

        for key, col in docs_vec_cols.items():
            if col is not None:
                col_doc_type = cls.doc_type._get_field_type(key).doc_type
                col_ = ListAdvancedIndexing(
                    cls.__class_getitem__(col_doc_type)._from_json_col_dict(
                        vec, tensor_type=tensor_type
                    )
                    for vec in col
                )
                docs_vec_cols[key] = col_
            else:
                docs_vec_cols[key] = None

        for key, col in any_cols.items():
            if col is not None:
                col_type = cls.doc_type._get_field_type(key)
                col_type = (
                    col_type
                    if cls.doc_type.__fields__[key].required
                    else Optional[col_type]
                )
                col_ = ListAdvancedIndexing(parse_obj_as(col_type, val) for val in col)
                any_cols[key] = col_
            else:
                any_cols[key] = None

        return cls.from_columns_storage(
            ColumnStorage(
                tensor_cols, doc_cols, docs_vec_cols, any_cols, tensor_type=tensor_type
            )
        )

    @classmethod
    def from_protobuf(
        cls: Type[T], pb_msg: 'DocVecProto', tensor_type: Type[AbstractTensor] = NdArray  # type: ignore
    ) -> T:
        """create a DocVec from a protobuf message
        :param pb_msg: the protobuf message to deserialize
        :param tensor_type: the tensor type to use for the tensor columns.
            Could be NdArray, TorchTensor, or TensorFlowTensor. Defaults to NdArray.
            All tensors of the output DocVec will be of this type.
        :return: The deserialized DocVec
        """
        tensor_columns: Dict[str, Optional[AbstractTensor]] = {}
        doc_columns: Dict[str, Optional['DocVec']] = {}
        docs_vec_columns: Dict[str, Optional[ListAdvancedIndexing['DocVec']]] = {}
        any_columns: Dict[str, ListAdvancedIndexing] = {}

        for tens_col_name, tens_col_proto in pb_msg.tensor_columns.items():
            if _is_none_ndarray_proto(tens_col_proto):
                # handle values that were None before serialization
                tensor_columns[tens_col_name] = None
            else:
                tensor_columns[tens_col_name] = tensor_type.from_protobuf(
                    tens_col_proto
                )

        for doc_col_name, doc_col_proto in pb_msg.doc_columns.items():
            if _is_none_docvec_proto(doc_col_proto):
                # handle values that were None before serialization
                doc_columns[doc_col_name] = None
            else:
                col_doc_type: Type = cls.doc_type._get_field_type(doc_col_name)
                doc_columns[doc_col_name] = cls.__class_getitem__(
                    col_doc_type
                ).from_protobuf(doc_col_proto, tensor_type=tensor_type)

        for docs_vec_col_name, docs_vec_col_proto in pb_msg.docs_vec_columns.items():
            vec_list: Optional[ListAdvancedIndexing]
            if _is_none_list_of_docvec_proto(docs_vec_col_proto):
                # handle values that were None before serialization
                vec_list = None
            else:
                vec_list = ListAdvancedIndexing()
                for doc_list_proto in docs_vec_col_proto.data:
                    col_doc_type = cls.doc_type._get_field_type(
                        docs_vec_col_name
                    ).doc_type
                    vec_list.append(
                        cls.__class_getitem__(col_doc_type).from_protobuf(
                            doc_list_proto, tensor_type=tensor_type
                        )
                    )
            docs_vec_columns[docs_vec_col_name] = vec_list

        for any_col_name, any_col_proto in pb_msg.any_columns.items():
            any_column: ListAdvancedIndexing = ListAdvancedIndexing()
            for node_proto in any_col_proto.data:
                content = cls.doc_type._get_content_from_node_proto(
                    node_proto, any_col_name
                )
                any_column.append(content)
            any_columns[any_col_name] = any_column

        storage = ColumnStorage(
            tensor_columns=tensor_columns,
            doc_columns=doc_columns,
            docs_vec_columns=docs_vec_columns,
            any_columns=any_columns,
            tensor_type=tensor_type,
        )

        return cls.from_columns_storage(storage)

    def to_protobuf(self) -> 'DocVecProto':
        """Convert DocVec into a Protobuf message"""
        from docarray.proto import (
            DocVecProto,
            ListOfAnyProto,
            ListOfDocArrayProto,
            ListOfDocVecProto,
            NdArrayProto,
        )

        self_ = cast('DocVec', self)

        doc_columns_proto: Dict[str, DocVecProto] = dict()
        tensor_columns_proto: Dict[str, NdArrayProto] = dict()
        da_columns_proto: Dict[str, ListOfDocArrayProto] = dict()
        any_columns_proto: Dict[str, ListOfAnyProto] = dict()

        for field, col_doc in self_._storage.doc_columns.items():
            if col_doc is None:
                # put dummy empty DocVecProto for serialization
                doc_columns_proto[field] = _none_docvec_proto()
            else:
                doc_columns_proto[field] = col_doc.to_protobuf()
        for field, col_tens in self_._storage.tensor_columns.items():
            if col_tens is None:
                # put dummy empty NdArrayProto for serialization
                tensor_columns_proto[field] = _none_ndarray_proto()
            else:
                tensor_columns_proto[field] = (
                    col_tens.to_protobuf() if col_tens is not None else None
                )
        for field, col_da in self_._storage.docs_vec_columns.items():
            list_proto = ListOfDocVecProto()
            if col_da:
                for docs in col_da:
                    list_proto.data.append(docs.to_protobuf())
            else:
                # put dummy empty ListOfDocVecProto for serialization
                list_proto = _none_list_of_docvec_proto()
            da_columns_proto[field] = list_proto
        for field, col_any in self_._storage.any_columns.items():
            list_proto = ListOfAnyProto()
            for data in col_any:
                list_proto.data.append(_type_to_protobuf(data))
            any_columns_proto[field] = list_proto

        return DocVecProto(
            doc_columns=doc_columns_proto,
            tensor_columns=tensor_columns_proto,
            docs_vec_columns=da_columns_proto,
            any_columns=any_columns_proto,
        )

    def to_csv(
        self, file_path: str, dialect: Union[str, 'csv.Dialect'] = 'excel'
    ) -> None:
        """
        DocVec does not support `.to_csv()`. This is because CSV is a row-based format
        while DocVec has a column-based data layout.
        To overcome this, do: `doc_vec.to_doc_list().to_csv(...)`.
        """
        raise NotImplementedError(
            f'{type(self)} does not support `.to_csv()`. This is because CSV is a row-based format'
            f'while {type(self)} has a column-based data layout. '
            f'To overcome this, do: `doc_vec.to_doc_list().to_csv(...)`.'
        )

    @classmethod
    def from_csv(
        cls: Type['T'],
        file_path: str,
        encoding: str = 'utf-8',
        dialect: Union[str, 'csv.Dialect'] = 'excel',
    ) -> 'T':
        """
        DocVec does not support `.from_csv()`. This is because CSV is a row-based format
        while DocVec has a column-based data layout.
        To overcome this, do: `DocList[MyDoc].from_csv(...).to_doc_vec()`.
        """
        raise NotImplementedError(
            f'{cls} does not support `.from_csv()`. This is because CSV is a row-based format while'
            f'{cls} has a column-based data layout. '
            f'To overcome this, do: `DocList[MyDoc].from_csv(...).to_doc_vec()`.'
        )

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> T:
        """Deserialize base64 strings into a `DocVec`.

        :param data: Base64 string to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compress algorithm that was used to serialize between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param tensor_type: the tensor type of the resulting DocVEc
        :return: the deserialized `DocVec`
        """
        return cls._load_binary_all(
            file_ctx=nullcontext(base64.b64decode(data)),
            protocol=protocol,
            compress=compress,
            show_progress=show_progress,
            tensor_type=tensor_type,
        )

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> T:
        """Deserialize bytes into a `DocList`.

        :param data: Bytes from which to deserialize
        :param protocol: protocol that was used to serialize
        :param compress: compression algorithm that was used to serialize between `lz4`, `bz2`, `lzma`, `zlib`, `gzip`
        :param show_progress: show progress bar, only works when protocol is `pickle` or `protobuf`
        :param tensor_type: the tensor type of the resulting DocVec
        :return: the deserialized `DocVec`
        """
        return cls._load_binary_all(
            file_ctx=nullcontext(data),
            protocol=protocol,
            compress=compress,
            show_progress=show_progress,
            tensor_type=tensor_type,
        )

    @classmethod
    def from_dataframe(
        cls: Type['T'],
        df: 'pd.DataFrame',
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> 'T':
        """
        Load a `DocVec` from a `pandas.DataFrame` following the schema
        defined in the [`.doc_type`][docarray.DocVec] attribute.
        Every row of the dataframe will be mapped to one Document in the doc_vec.
        The column names of the dataframe have to match the field names of the
        Document type.
        For nested fields use "__"-separated access paths as column names,
        such as `'image__url'`.

        List-like fields (including field of type DocList) are not supported.

        ---

        ```python
        import pandas as pd

        from docarray import BaseDoc, DocVec


        class Person(BaseDoc):
            name: str
            follower: int


        df = pd.DataFrame(
            data=[['Maria', 12345], ['Jake', 54321]], columns=['name', 'follower']
        )

        docs = DocVec[Person].from_dataframe(df)

        assert docs.name == ['Maria', 'Jake']
        assert docs.follower == [12345, 54321]
        ```

        ---

        :param df: `pandas.DataFrame` to extract Document's information from
        :param tensor_type: the tensor type of the resulting DocVec
        :return: `DocList` where each Document contains the information of one
            corresponding row of the `pandas.DataFrame`.
        """
        # type ignore could be avoided by simply putting this implementation in the DocVec class
        # but leaving it here for code separation
        return cls(super().from_dataframe(df), tensor_type=tensor_type)  # type: ignore

    @classmethod
    def load_binary(
        cls: Type[T],
        file: Union[str, bytes, pathlib.Path, io.BufferedReader, _LazyRequestReader],
        protocol: str = 'protobuf-array',
        compress: Optional[str] = None,
        show_progress: bool = False,
        streaming: bool = False,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> Union[T, Generator['T_doc', None, None]]:
        """Load doc_vec elements from a compressed binary file.

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
        :param tensor_type: the tensor type of the resulting DocVEc

        :return: a `DocVec` object

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
                file_ctx,
                load_protocol,
                load_compress,
                show_progress,
                tensor_type=tensor_type,
            )
