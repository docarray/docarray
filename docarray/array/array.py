from collections import defaultdict
from functools import wraps
from typing import Dict, Iterable, List, Optional, Type

import torch

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.array.mixins import GetAttributeArrayMixin, ProtoArrayMixin
from docarray.document import AnyDocument, BaseDocument, BaseNode
from docarray.typing import TorchTensor


def _stack_mode_blocker(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_stack():
            raise RuntimeError(
                f'Cannot call {func.__name__} when the document array is in stack mode'
            )
        return func(self, *args, **kwargs)

    return wrapper


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

        self._tensor_columns: Optional[Dict[str, Optional[TorchTensor]]] = dict()

        for field_name, field in self.document_type.__fields__.items():
            if issubclass(field.type_, TorchTensor):
                self._tensor_columns[field_name] = None

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'DocumentArray[item] item should be a Document not a {item} '
            )

        class _DocumenArrayTyped(cls):  # type: ignore
            document_type: Type[BaseDocument] = item

        for field in _DocumenArrayTyped.document_type.__fields__.keys():

            def _property_generator(val: str):
                return property(lambda self: self._get_documents_attribute(val))

            setattr(_DocumenArrayTyped, field, _property_generator(field))
            # this generates property on the fly based on the schema of the item

        _DocumenArrayTyped.__name__ = f'DocumentArray[{item.__name__}]'
        _DocumenArrayTyped.__qualname__ = f'DocumentArray[{item.__name__}]'

        return _DocumenArrayTyped

    def stacked(self):

        if not (self.is_stack()):
            tensors_to_stack: Dict[str, List[TorchTensor]] = defaultdict(list)
            for doc in self:
                for tensor_field in self._tensor_columns.keys():
                    tensors_to_stack[tensor_field].append(getattr(doc, tensor_field))
                    setattr(doc, tensor_field, None)

            for tensor_field, to_stack in tensors_to_stack.items():
                self._tensor_columns[tensor_field] = torch.stack(to_stack)

            for i, doc in enumerate(self):
                for tensor_field in self._tensor_columns.keys():
                    setattr(doc, tensor_field, self._tensor_columns[tensor_field][i])

    def unstacked(self):
        if self.is_stack():

            for field in self._tensor_columns.keys():
                del self._tensor_columns[field]

            self._tensor_columns = dict()

    def is_stack(self) -> bool:
        return self._tensor_columns != dict()

    append = _stack_mode_blocker(list.append)
    extend = _stack_mode_blocker(list.extend)
    clear = _stack_mode_blocker(list.clear)
    insert = _stack_mode_blocker(list.insert)
    pop = _stack_mode_blocker(list.pop)
    remove = _stack_mode_blocker(list.remove)
    reverse = _stack_mode_blocker(list.reverse)
    sort = _stack_mode_blocker(list.sort)
