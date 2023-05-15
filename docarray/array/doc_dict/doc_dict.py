from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from typing_inspect import is_union_type

from docarray.array.any_collections import AnyCollections
from docarray.array.doc_list.doc_list import DocList
from docarray.base_doc import AnyDoc
from docarray.base_doc.doc import BaseDoc

if TYPE_CHECKING:
    from docarray.typing import AbstractTensor, NdArray, TorchTensor

T_doc = TypeVar('T_doc', bound=BaseDoc)
T = TypeVar('T', bound='DocDict')


class DocDict(AnyCollections[T_doc]):
    doc_type: Type[BaseDoc] = AnyDoc

    def __init__(self, **mapping):
        self._data = {key: self._validate_one_doc(val) for key, val in mapping.items()}

    def _validate_one_doc(self, doc: T_doc) -> T_doc:
        """Validate if a Document is compatible with this `DocList`"""
        if not issubclass(self.doc_type, AnyDoc) and not isinstance(doc, self.doc_type):
            raise ValueError(f'{doc} is not a {self.doc_type}')
        return doc

    @classmethod
    def from_doc_list(cls: Type[T], docs: DocList) -> T:
        return cls(**{doc.id: doc for doc in docs})

    def __iter__(self):
        for key in self._data.keys():
            yield key

    def update(self, other: Union['DocDict', Dict[str, T_doc], DocList]):
        if isinstance(other, DocDict):
            self._data.update(other._data)
        elif isinstance(other, DocList):
            self._data.update(DocDict.from_doc_list(other)._data)
        else:
            self._data.update(other)

    def __getitem__(self, item):
        return self._data[item]

    def items(self):
        return self._data.items()

    def _get_data_column(
        self: T,
        field: str,
    ) -> Union[T, Dict[Any, Union[BaseDoc, 'TorchTensor', 'NdArray']]]:

        field_type = self.__class__.doc_type._get_field_type(field)

        if (
            not is_union_type(field_type)
            and self.__class__.doc_type.__fields__[field].required
            and isinstance(field_type, type)
            and issubclass(field_type, BaseDoc)
        ):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return DocDict.__class_getitem__(field_type)(
                {key: getattr(doc, field) for key, doc in self.items()},
            )
        else:
            return {key: getattr(doc, field) for key, doc in self.items()}

    def _set_data_column(
        self: T,
        field: str,
        values: Union[List, T, 'AbstractTensor'],
    ):
        """Set all Documents in this `DocList` using the passed values

        :param field: name of the fields to set
        :values: the values to set at the `DocList` level
        """
        raise NotImplementedError

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg) -> T:
        """create a Document from a protobuf message"""
        raise NotImplementedError

    def to_protobuf(self):
        """Convert DocList into a Protobuf message"""
        raise NotImplementedError
