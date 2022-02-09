from abc import ABC

from .content import ContentPropertyMixin
from .delitem import DelItemMixin
from .embed import EmbedMixin
from .empty import EmptyMixin
from .evaluation import EvaluationMixin
from .getattr import GetAttributeMixin
from .getitem import GetItemMixin
from .group import GroupMixin
from .io.binary import BinaryIOMixin
from .io.common import CommonIOMixin
from .io.csv import CsvIOMixin
from .io.dataframe import DataframeIOMixin
from .io.from_gen import FromGeneratorMixin
from .io.json import JsonIOMixin
from .io.pushpull import PushPullMixin
from .match import MatchMixin
from .parallel import ParallelMixin
from .plot import PlotMixin
from .pydantic import PydanticMixin
from .reduce import ReduceMixin
from .sample import SampleMixin
from .setitem import SetItemMixin
from .strawberry import StrawberryMixin
from .text import TextToolsMixin
from .traverse import TraverseMixin


class AllMixins(
    GetAttributeMixin,
    GetItemMixin,
    SetItemMixin,
    DelItemMixin,
    ContentPropertyMixin,
    PydanticMixin,
    StrawberryMixin,
    GroupMixin,
    EmptyMixin,
    CsvIOMixin,
    JsonIOMixin,
    BinaryIOMixin,
    CommonIOMixin,
    EmbedMixin,
    PushPullMixin,
    FromGeneratorMixin,
    MatchMixin,
    TraverseMixin,
    PlotMixin,
    SampleMixin,
    TextToolsMixin,
    EvaluationMixin,
    ReduceMixin,
    ParallelMixin,
    DataframeIOMixin,
    ABC,
):
    """All plugins that can be used in :class:`DocumentArray`."""

    ...
