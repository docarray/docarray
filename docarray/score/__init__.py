from docarray.score.data import NamedScoreData
from docarray.score.mixins import AllMixins
from docarray.base import BaseDCType


class NamedScore(AllMixins, BaseDCType):
    _data_class = NamedScoreData
    _post_init_fields = ()
