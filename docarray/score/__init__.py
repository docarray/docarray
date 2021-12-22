from ..base import BaseDCType
from .data import NamedScoreData
from .mixins import AllMixins


class NamedScore(AllMixins, BaseDCType):
    _data_class = NamedScoreData
