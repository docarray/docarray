from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    Type,
    TypeVar,
    Union,
    cast,
)

from docarray.base_doc import BaseDoc
from docarray.typing.abstract_type import AbstractType
from docarray.utils._internal._typing import change_cls_name

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import DocListProto, NodeProto

T = TypeVar('T', bound='AnyCollections')
T_doc = TypeVar('T_doc', bound=BaseDoc)
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


class AnyCollections(Generic[T_doc], AbstractType):
    doc_type: Type[BaseDoc]
    __typed_da__: Dict[Type['AnyCollections'], Dict[Type[BaseDoc], Type]] = {}

    def __repr__(self):
        return f'<{self.__class__.__name__} (length={len(self)})>'

    @classmethod
    def __class_getitem__(cls, item: Union[Type[BaseDoc], TypeVar, str]):
        if not isinstance(item, type):
            return Generic.__class_getitem__.__func__(cls, item)  # type: ignore
            # this do nothing that checking that item is valid type var or str
        if not issubclass(item, BaseDoc):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        if cls not in cls.__typed_da__:
            cls.__typed_da__[cls] = {}

        if item not in cls.__typed_da__[cls]:
            # Promote to global scope so multiprocessing can pickle it
            global _DocArrayTyped

            class _DocArrayTyped(cls):  # type: ignore
                doc_type: Type[BaseDoc] = cast(Type[BaseDoc], item)

            for field in _DocArrayTyped.doc_type.__fields__.keys():

                def _property_generator(val: str):
                    def _getter(self):
                        return self._get_data_column(val)

                    def _setter(self, value):
                        self._set_data_column(val, value)

                    # need docstring for the property
                    return property(fget=_getter, fset=_setter)

                setattr(_DocArrayTyped, field, _property_generator(field))
                # this generates property on the fly based on the schema of the item

            # The global scope and qualname need to refer to this class a unique name.
            # Otherwise, creating another _DocArrayTyped will overwrite this one.
            change_cls_name(
                _DocArrayTyped, f'{cls.__name__}[{item.__name__}]', globals()
            )

            cls.__typed_da__[cls][item] = _DocArrayTyped

        return cls.__typed_da__[cls][item]

    def __getattr__(self, item: str):
        # Needs to be explicitly defined here for the purpose to disable PyCharm's complaints
        # about not detected properties: https://youtrack.jetbrains.com/issue/PY-47991
        return super().__getattribute__(item)

    @abstractmethod
    def _get_data_column(
        self: T,
        field: str,
    ):
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        ...

    @abstractmethod
    def _set_data_column(
        self: T,
        field: str,
        values: Any,
    ):
        """Set all Documents in this [`DocList`][docarray.array.doc_list.doc_list.DocList] using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocList level
        """
        ...

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocListProto') -> T:
        """create a Document from a protobuf message"""
        ...

    @abstractmethod
    def to_protobuf(self) -> 'DocListProto':
        """Convert DocList into a Protobuf message"""
        ...

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert a [`DocList`][docarray.array.doc_list.doc_list.DocList] into a NodeProto
        protobuf message.
        This function should be called when a DocList is nested into
        another Document that need to be converted into a protobuf.

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(doc_array=self.to_protobuf())

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        else:
            raise ValueError(f'Value {value} is not a valid DocDict type')
