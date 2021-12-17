from docarray.base import BaseDCType
from .data import NamedScoreData
from .mixin import AllMixins


class NamedScore(AllMixins, BaseDCType):
    _data_class = NamedScoreData
