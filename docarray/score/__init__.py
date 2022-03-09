from .data import NamedScoreData
from .mixins import AllMixins
from ..base import BaseDCType


class NamedScore(AllMixins, BaseDCType):
    _data_class = NamedScoreData
    _post_init_fields = ()
