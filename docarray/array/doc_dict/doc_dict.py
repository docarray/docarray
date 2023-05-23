from typing import TYPE_CHECKING, Any, Dict, Mapping, Type, TypeVar, Union

from typing_inspect import is_union_type

from docarray.array.any_collections import AnyCollection
from docarray.array.doc_list.doc_list import DocList
from docarray.base_doc import AnyDoc
from docarray.base_doc.doc import BaseDoc

if TYPE_CHECKING:
    from docarray.typing import NdArray, TorchTensor

T_doc = TypeVar('T_doc', bound=BaseDoc)
T = TypeVar('T', bound='DocDict')


class DocDict(AnyCollection[T_doc], Dict[str, T_doc]):
    doc_type: Type[BaseDoc] = AnyDoc

    def __init__(self, **mapping):
        super().__init__(
            {key: self._validate_one_doc(val) for key, val in mapping.items()}
        )

    def _validate_one_doc(self, doc: T_doc) -> T_doc:
        """Validate if a Document is compatible with this `DocList`"""
        if not issubclass(self.doc_type, AnyDoc) and not isinstance(doc, self.doc_type):
            raise ValueError(f'{doc} is not a {self.doc_type}')
        return doc

    @classmethod
    def from_doc_list(cls: Type[T], docs: DocList) -> T:
        return cls(**{doc.id: doc for doc in docs})

    # here we need to ignore type as the DocDict as a signature incompatible with Dict (it is more restrictive)
    def update(self, other: Union['DocDict', Dict[str, T_doc], DocList]):  # type: ignore
        if isinstance(other, DocDict):
            super().update(other)
        elif isinstance(other, DocList):
            super().update(DocDict.from_doc_list(other))
        else:
            super().update(other)

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
            return DocDict.__class_getitem__(field_type)(  # todo skip validation
                **{key: getattr(doc, field) for key, doc in self.items()},
            )
        else:
            return {key: getattr(doc, field) for key, doc in self.items()}

    def _set_data_column(
        self: T,
        field: str,
        values: Union[Mapping[str, Any]],
    ):
        """Set all Documents in this `DocDict` using the passed values

        :param field: name of the fields to set
        :values: the values to set at the `DocList` level
        """
        if values.keys() != self.keys():
            raise ValueError(
                f'Keys of the values {values.keys()} do not match the keys of the DocDict {self.keys()}'
            )

        for key, value in values.items():
            setattr(self[key], field, value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg) -> T:
        """create a Document from a protobuf message"""
        raise NotImplementedError

    def to_protobuf(self):
        """Convert DocList into a Protobuf message"""
        raise NotImplementedError
