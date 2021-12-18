from .data import DocumentData, default_values
from .mixins import AllMixins
from ..base import BaseDCType


class Document(AllMixins, BaseDCType):
    _data_class = DocumentData
    _unresolved_fields_dest = 'tags'
