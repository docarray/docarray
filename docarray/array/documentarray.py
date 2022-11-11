from typing import Iterable, Type

from docarray.document import AnyDocument, BaseDocument
from docarray.document.abstract_document import AbstractDocument
from docarray.typing import BaseNode

from .abstract_array import AbstractDocumentArray
from .mixins import ProtoArrayMixin


class DocumentArray(
    list,
    ProtoArrayMixin,
    AbstractDocumentArray,
    BaseNode,
):
    """
    a _GenericDocumentArray is a list-like container of Document of the same schema

    :param docs: iterable of Document
    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(self, docs: Iterable[AbstractDocument]):
        super().__init__(doc_ for doc_ in docs)

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'DocumentArray[item] item should be a Document not a {item} '
            )

        class _DocumenArrayTyped(DocumentArray):
            document_type = item

        _DocumenArrayTyped.__name__ = f'DocumentArray{item.__name__}'

        return _DocumenArrayTyped
