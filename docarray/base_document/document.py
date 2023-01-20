import os
from typing import TYPE_CHECKING, Any, Optional, Type

import numpy as np
import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.tree import Tree

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import ProtoMixin
from docarray.math.helper import minmax_normalize
from docarray.typing import ID

if TYPE_CHECKING:
    # import colorsys
    # from typing import Any, Optional, TypeVar
    # import numpy as np
    # from rich.color import Color
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement

    # from rich.segment import Segment
    # from rich.style import Style
    # from rich.tree import Tree
    #
    # import docarray
    # from docarray.math.helper import minmax_normalize
    # from docarray.typing import ID


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def _get_field_type(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].outer_type_

    def _ipython_display_(self):
        """Displays the object in IPython as a side effect"""
        self.summary()

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from rich import print

        t = _plot_recursion(node=self)
        print(t)

    def schema_summary(self) -> None:
        from rich import print
        from rich.panel import Panel

        panel = Panel(
            self.get_schema(), title='Document Schema', expand=False, padding=(1, 3)
        )
        print(panel)

    def get_schema(self, doc_name: str = None) -> Tree:
        from rich.tree import Tree

        n = self.__class__.__name__

        tree = Tree(n) if doc_name is None else Tree(f'{doc_name}: {n}')
        annotations = self.__annotations__
        for k, v in annotations.items():
            value = getattr(self, k)
            if isinstance(value, BaseDocument):
                tree.add(value.get_schema(doc_name=k))
            else:
                t = str(v).replace('[', '\[')
                import re

                t = re.sub('[a-zA-Z_]*[.]', '', t)
                if 'Union' in t and 'NoneType' in t:
                    t = t.replace('Union', 'Optional').replace(', NoneType', '')
                tree.add(f'{k}: {t}')
        return tree

    def __rich_console__(self, console, options):
        kls = self.__class__.__name__
        id_abbrv = getattr(self, 'id')[:7]
        yield f":page_facing_up: [b]{kls}" f"[/b]: [cyan]{id_abbrv} ...[cyan]"

        from collections.abc import Iterable

        import torch
        from rich import box, text
        from rich.table import Table

        my_table = Table(
            'Attribute', 'Value', width=80, box=box.ROUNDED, highlight=True
        )

        for k, v in self.__dict__.items():
            col_1, col_2 = '', ''

            if isinstance(v, ID) or k.startswith('_') or v is None:
                continue
            elif isinstance(v, str):
                col_1 = f'{k}: {v.__class__.__name__}'
                col_2 = str(v)[:50]
                if len(v) > 50:
                    col_2 += f' ... (length: {len(v)})'
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                col_1 = f'{k}: {v.__class__.__name__}'

                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                if v.squeeze().ndim == 1 and len(v) < 1000:
                    col_2 = ColorBoxArray(v.squeeze())
                else:
                    col_2 = f'{type(v)} of shape {v.shape}, dtype: {v.dtype}'

            elif isinstance(v, tuple) or isinstance(v, list):
                col_1 = f'{k}: {v.__class__.__name__}'
                for i, x in enumerate(v):
                    if len(col_2) + len(str(x)) < 50:
                        col_2 = str(v[:i])
                    else:
                        col_2 = f'{col_2[:-1]}, ...] (length: {len(v)})'
                        break
            elif not isinstance(v, Iterable):
                col_1 = f'{k}: {v.__class__.__name__}'
                col_2 = str(v)
            else:
                continue

            if not isinstance(col_2, ColorBoxArray):
                col_2 = text.Text(col_2)
            my_table.add_row(col_1, col_2)

        if my_table.rows:
            yield my_table


def _plot_recursion(node: Any, tree: Optional[Tree] = None) -> Tree:
    import docarray

    tree = Tree(node) if tree is None else tree.add(node)

    try:
        iterable_attrs = [
            k
            for k, v in node.__dict__.items()
            if isinstance(v, docarray.DocumentArray)
            or isinstance(v, docarray.BaseDocument)
        ]
        for attr in iterable_attrs:
            _icon = ':diamond_with_a_dot:'
            value = getattr(node, attr)
            if isinstance(value, docarray.BaseDocument):
                _icon = ':large_orange_diamond:'
            _match_tree = tree.add(
                f'{_icon} [b]{attr}: ' f'{value.__class__.__name__}[/b]'
            )
            if isinstance(value, docarray.BaseDocument):
                value = [value]
            for i, d in enumerate(value):
                if i == 2:
                    _plot_recursion(
                        f'... {len(value) - 2} more {d.__class__.__name__} documents\n',
                        _match_tree,
                    )
                    break
                _plot_recursion(d, _match_tree)

    except Exception:
        pass

    return tree


class ColorBoxArray:
    def __init__(self, array):
        self._array = minmax_normalize(array, (0, 5))

    def __rich_console__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'RenderResult':
        import colorsys

        from rich.color import Color
        from rich.segment import Segment
        from rich.style import Style

        h = 0.75
        for idx, y in enumerate(self._array):
            lightness = 0.1 + ((y / 5) * 0.7)
            r, g, b = colorsys.hls_to_rgb(h, lightness + 0.7 / 10, 1.0)
            color = Color.from_rgb(r * 255, g * 255, b * 255)
            yield Segment('▄', Style(color=color, bgcolor=color))
            if idx != 0 and idx % options.max_width == 0:
                yield Segment.line()

    def __rich_measure__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'Measurement':
        from rich.measure import Measurement

        return Measurement(1, options.max_width)
