import colorsys
from typing import Any, Optional, TypeVar

import numpy as np
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from rich.tree import Tree

import docarray
from docarray.math.helper import minmax_normalize
from docarray.typing import ID

T = TypeVar('T', bound=Any)


class PlotMixin:
    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from rich import print

        t = PlotMixin._plot_recursion(node=self)
        print(t)

    @staticmethod
    def _plot_recursion(node: T, tree: Optional[Tree] = None) -> Tree:
        tree = Tree(node) if tree is None else tree.add(node)

        try:
            iterable_attrs = [
                k
                for k, v in node.__dict__.items()
                if isinstance(v, docarray.DocumentArray)
            ]
            for attr in iterable_attrs:
                value = getattr(node, attr)
                _icon = ':diamond_with_a_dot:'
                _match_tree = tree.add(
                    f'{_icon} [b]{attr.capitalize()}: '
                    f'{value.__class__.__name__}[/b]'
                )
                for i, d in enumerate(value):
                    if i == 2:
                        PlotMixin._plot_recursion(
                            f' ... {len(value) - 2} more {d.__class__} documents',
                            _match_tree,
                        )
                        break
                    PlotMixin._plot_recursion(d, _match_tree)

        except Exception:
            pass

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
                    v = v.detach().cou().numpy()
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


class ColorBoxArray:
    def __init__(self, array):
        self._array = minmax_normalize(array, (0, 5))

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        h = 0.75
        for idx, y in enumerate(self._array):
            lightness = 0.1 + ((y / 5) * 0.7)
            r, g, b = colorsys.hls_to_rgb(h, lightness + 0.7 / 10, 1.0)
            color = Color.from_rgb(r * 255, g * 255, b * 255)
            yield Segment('▄', Style(color=color, bgcolor=color))
            if idx != 0 and idx % options.max_width == 0:
                yield Segment.line()

    def __rich_measure__(
        self, console: "Console", options: ConsoleOptions
    ) -> Measurement:
        return Measurement(1, options.max_width)
