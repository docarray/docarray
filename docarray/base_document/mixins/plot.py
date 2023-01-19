from typing import Any, Optional, TypeVar

import numpy as np
from rich.tree import Tree

import docarray
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
                            f' ... {len(value) - 2} more Docs', _match_tree
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

            if k.startswith('_') or isinstance(v, ID) or v is None:
                continue
            elif isinstance(v, str):
                col_1 = f'{k}: {v.__class__.__name__}'
                col_2 = str(v)[:50]
                if len(v) > 50:
                    col_2 += f' ... (length: {len(v)})'
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                col_1 = f'{k}: {v.__class__.__name__}'
                col_2 = f'{type(v)} in shape {v.shape}, dtype: {v.dtype}'
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

            my_table.add_row(col_1, text.Text(col_2))

        if my_table.rows:
            yield my_table
