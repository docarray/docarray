from abc import ABC

from docarray.array.mixins.content import ContentPropertyMixin
from docarray.array.mixins.delitem import DelItemMixin
from docarray.array.mixins.embed import EmbedMixin
from docarray.array.mixins.empty import EmptyMixin
from docarray.array.mixins.evaluation import EvaluationMixin
from docarray.array.mixins.find import FindMixin
from docarray.array.mixins.getattr import GetAttributeMixin
from docarray.array.mixins.getitem import GetItemMixin
from docarray.array.mixins.group import GroupMixin
from docarray.array.mixins.io.binary import BinaryIOMixin
from docarray.array.mixins.io.common import CommonIOMixin
from docarray.array.mixins.io.csv import CsvIOMixin
from docarray.array.mixins.io.dataframe import DataframeIOMixin
from docarray.array.mixins.io.from_gen import FromGeneratorMixin
from docarray.array.mixins.io.json import JsonIOMixin
from docarray.array.mixins.io.pushpull import PushPullMixin
from docarray.array.mixins.match import MatchMixin
from docarray.array.mixins.parallel import ParallelMixin
from docarray.array.mixins.plot import PlotMixin
from docarray.array.mixins.post import PostMixin
from docarray.array.mixins.pydantic import PydanticMixin
from docarray.array.mixins.reduce import ReduceMixin
from docarray.array.mixins.sample import SampleMixin
from docarray.array.mixins.setitem import SetItemMixin
from docarray.array.mixins.strawberry import StrawberryMixin
from docarray.array.mixins.text import TextToolsMixin
from docarray.array.mixins.traverse import TraverseMixin
from docarray.array.mixins.dataloader import DataLoaderMixin


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
    FindMixin,
    MatchMixin,
    TraverseMixin,
    PlotMixin,
    SampleMixin,
    PostMixin,
    TextToolsMixin,
    EvaluationMixin,
    ReduceMixin,
    ParallelMixin,
    DataframeIOMixin,
    DataLoaderMixin,
    ABC,
):
    """All plugins that can be used in :class:`DocumentArray`."""

    ...
