from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Iterable, List, Optional, Type, TypeVar, Union

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.array.mixins import GetAttributeArrayMixin, ProtoArrayMixin
from docarray.document import AnyDocument, BaseDocument, BaseNode
from docarray.typing import NdArray, TorchTensor


def _stacked_mode_blocker(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_stacked():
            raise RuntimeError(
                f'Cannot call {func.__name__} when the document array is in stack mode'
            )
        return func(self, *args, **kwargs)

    return wrapper


T = TypeVar('T', bound='DocumentArray')


class DocumentArray(
    list,
    ProtoArrayMixin,
    GetAttributeArrayMixin,
    AbstractDocumentArray,
    BaseNode,
):
    """
    a DocumentArray is a list-like container of Document of the same schema

    :param docs: iterable of Document
    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(self, docs: Iterable[BaseDocument]):
        super().__init__(doc_ for doc_ in docs)

        self._columns: Optional[
            Dict[str, Union[TorchTensor, AbstractDocumentArray, NdArray, None]]
        ] = None

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'DocumentArray[item] item should be a Document not a {item} '
            )

        class _DocumentArrayTyped(cls):  # type: ignore
            document_type: Type[BaseDocument] = item

        for field in _DocumentArrayTyped.document_type.__fields__.keys():

            def _property_generator(val: str):
                def _getter(self):
                    return self._get_array_attribute(val)

                def _setter(self, value):
                    self._set_array_attribute(val, value)

                # need docstring for the property
                return property(fget=_getter, fset=_setter)

            setattr(_DocumentArrayTyped, field, _property_generator(field))
            # this generates property on the fly based on the schema of the item

        _DocumentArrayTyped.__name__ = f'DocumentArray[{item.__name__}]'
        _DocumentArrayTyped.__qualname__ = f'DocumentArray[{item.__name__}]'

        return _DocumentArrayTyped

    def __getitem__(self, item):
        if self.is_stacked():
            doc = super().__getitem__(item)
            # NOTE: this could be speed up by using a cache
            for field in self._columns.keys():
                setattr(doc, field, self._columns[field][item])
            return doc
        else:
            return super().__getitem__(item)

    def stack(self):

        if not (self.is_stacked()):

            self._columns = dict()

            for field_name, field in self.document_type.__fields__.items():
                if (
                    issubclass(field.type_, TorchTensor)
                    or issubclass(field.type_, BaseDocument)
                    or issubclass(field.type_, NdArray)
                ):
                    self._columns[field_name] = None

            columns_to_stack: Dict[
                str, Union[List[TorchTensor], List[NdArray], List[BaseDocument]]
            ] = defaultdict(list)

            for doc in self:
                for field_to_stack in self._columns.keys():
                    columns_to_stack[field_to_stack].append(
                        getattr(doc, field_to_stack)
                    )
                    setattr(doc, field_to_stack, None)

            for field_to_stack, to_stack in columns_to_stack.items():
                type_ = self.document_type.__fields__[field_to_stack].type_
                if issubclass(type_, TorchTensor):
                    self._columns[field_to_stack] = TorchTensor.__docarray_stack__(
                        to_stack
                    )
                if issubclass(type_, NdArray):
                    self._columns[field_to_stack] = NdArray.__docarray_stack__(to_stack)
                elif issubclass(type_, BaseDocument):
                    self._columns[field_to_stack] = DocumentArray[type_](
                        to_stack
                    ).stack()

            for i, doc in enumerate(self):
                for field_to_stack in self._columns.keys():
                    type_ = self.document_type.__fields__[field_to_stack].type_
                    if issubclass(type_, TorchTensor) or issubclass(type_, NdArray):
                        setattr(doc, field_to_stack, self._columns[field_to_stack][i])

        return self

    @contextmanager
    def stacked_mode(self):
        try:
            yield self.stack()
        finally:
            self.unstack()

    @contextmanager
    def unstacked_mode(self):
        try:
            yield self.unstack()
        finally:
            self.stack()

    def unstack(self):
        if self.is_stacked():

            for field in list(self._columns.keys()):
                # list needed here otherwise we are modifying the dict while iterating
                del self._columns[field]

            self._columns = None

        return self

    def is_stacked(self) -> bool:
        return self._columns is not None

    append = _stacked_mode_blocker(list.append)
    extend = _stacked_mode_blocker(list.extend)
    clear = _stacked_mode_blocker(list.clear)
    insert = _stacked_mode_blocker(list.insert)
    pop = _stacked_mode_blocker(list.pop)
    remove = _stacked_mode_blocker(list.remove)
    reverse = _stacked_mode_blocker(list.reverse)
    sort = _stacked_mode_blocker(list.sort)
