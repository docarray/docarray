import typing
from typing import Optional

import numpy as np
from rich.tree import Tree

import docarray


class PlotMixin:
    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from rich import print

        t = self._plot_recursion()
        print(t)

    def _plot_recursion(self, tree: Optional[Tree] = None) -> Tree:
        if tree is None:
            tree = Tree(self)
        else:
            tree = tree.add(self)
        try:
            iterable_attrs = [
                x
                for x in self.__dict__.keys()
                if (
                    isinstance(getattr(self, x), typing.List)
                    or isinstance(getattr(self, x), typing.Tuple)
                    or isinstance(getattr(self, x), docarray.DocumentArray)
                )
            ]

            for attr in iterable_attrs:
                value = getattr(self, attr)
                if value:
                    _icon = ':diamond_with_a_dot:'
                    _match_tree = tree.add(
                        f'{_icon} [b]{attr.capitalize()}: '
                        f'{value.__class__.__name__}[/b]'
                    )
                    for d in value:
                        self._plot_recursion.__func__(d, _match_tree)
        except ():
            pass
        return tree

    def __rich_console__(self, console, options):
        print('in rich console')
        kls = self.__class__.__name__
        yield f":page_facing_up: [b]{kls}" f"[/b]: [cyan]{getattr(self, 'id')}[cyan]"
        from collections.abc import Iterable

        import torch
        from rich import box, text
        from rich.table import Table

        my_table = Table(
            'Attribute', 'Type', 'Value', width=80, box=box.ROUNDED, highlight=True
        )
        annotations = self.__annotations__

        print(f"annotations = {annotations}")
        for f, d in self.__dict__.items():
            v = getattr(self, f)
            if f.startswith('_') or f == 'id':
                continue
            elif isinstance(v, str):
                v_str = str(v)[:100]
                if len(v) > 100:
                    v_str += f'... (length: {len(v)})'
                my_table.add_row(f, 'string', text.Text(v_str))
            elif v is None:
                my_table.add_row(
                    f'{f}: {v.__class__.__name__}',
                    f'{v.__class__.__name__}',
                    text.Text('None'),
                )
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                x = f'{type(getattr(self, f))} in shape {v.shape}, dtype: {v.dtype}'
                my_table.add_row(
                    f'{f}: {v.__class__.__name__}', f'{v.__class__.__name__}', x
                )
            elif not isinstance(v, Iterable):
                my_table.add_row(
                    f'{f}: {v.__class__.__name__}',
                    f'{v.__class__.__name__}',
                    text.Text(str(getattr(self, f))),
                )
            elif isinstance(v, tuple) or isinstance(v, list):
                my_table.add_row(
                    f'{f}: {v.__class__.__name__}',
                    f'{v.__class__.__name__}',
                    text.Text(str(getattr(self, f))),
                )

        if my_table.rows:
            yield my_table
