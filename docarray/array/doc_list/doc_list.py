import io
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
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
from docarray.array.doc_list.io import IOMixinArray
from docarray.array.doc_list.pushpull import PushPullMixin
from docarray.array.doc_list.sequence_indexing_mixin import (
    IndexingSequenceMixin,
    IndexIterType,
)
from docarray.base_doc import AnyDoc, BaseDoc
from docarray.typing import NdArray

if TYPE_CHECKING:
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

    def to_doc_vec(
        self,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ) -> 'DocVec':
        """
        Convert the `DocList` into a `DocVec`. `Self` cannot be used
        afterward
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
