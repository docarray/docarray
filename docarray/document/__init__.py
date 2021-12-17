from .data import DocumentData
from .mixins import AllMixins
from ..base import BaseDCType


class Document(AllMixins, BaseDCType):
    _data_class = DocumentData
