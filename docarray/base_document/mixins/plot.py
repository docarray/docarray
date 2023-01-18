from typing import Optional

import numpy as np
from rich.tree import Tree


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
        print(f"tree.label = {tree.label}")
        print(f"tree.label.__class__.__name__ = {tree.label.__class__.__name__}")
        print(f"type(tree.label) = {type(tree.label)}")
        # tree.label = 'label'

        from collections.abc import Iterable

        iterable_attrs = [x for x in self.__dict__.keys() if isinstance(x, Iterable)]
        print(f"iterable_attrs = {iterable_attrs}")
        print(f"self.__dict__.keys() = {self.__dict__.keys()}")

        for attr in iterable_attrs:
            print(f"attr = {attr}")
            if getattr(self, attr):
                _icon = ':diamond_with_a_dot:'
                _match_tree = tree.add(f'{_icon} [b]{attr.capitalize()}[/b]')
                for d in getattr(self, attr):
                    d._plot_recursion(_match_tree)

        return tree

    def __rich_console__(self, console, options):

        yield f":page_facing_up: [b]Document[/b]: [cyan]{self.id}[cyan]"
        from collections.abc import Iterable

        import torch
        from rich import box, text
        from rich.table import Table

        my_table = Table(
            'Attribute', 'Value', width=80, box=box.ROUNDED, highlight=True
        )
        print(f"self.__dict__.keys() = {self.__dict__.keys()}")
        for f in self.__dict__.keys():
            print(f"f = {f}")
            v = getattr(self, f)
            print(f"v = {v}")
            print(f"isinstance(v, str) = {isinstance(v, str)}")
            if f.startswith('_') or f == 'id':
                continue
            elif isinstance(v, str):
                v_str = str(v)[:100]
                if len(v) > 100:
                    v_str += f'... (length: {len(v)})'
                my_table.add_row(f, text.Text(v_str))
            elif v is None:
                my_table.add_row(f, text.Text('None'))
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                x = f'{type(getattr(self, f))} in shape {v.shape}, dtype: {v.dtype}'
                my_table.add_row(f, x)
            elif not isinstance(v, Iterable):
                my_table.add_row(f, text.Text(str(getattr(self, f))))

        #     elif f in ('embedding', 'tensor'):
        #         from docarray.math.ndarray import to_numpy_array
        #
        #         v = to_numpy_array(getattr(self, f))
        #         if v.squeeze().ndim == 1 and len(v) < 1000:
        #             from docarray.document.mixins.rich_embedding import (
        #                 ColorBoxEmbedding,
        #             )
        #
        #             v = ColorBoxEmbedding(v.squeeze())
        #         else:
        #             v = f'{type(getattr(self, f))} shape {v.shape}, dtype: {v.dtype}'
        #         my_table.add_row(f, v)
        #     elif f not in ('id', 'chunks', 'matches'):
        #         my_table.add_row(f, Text(str(getattr(self, f))))
        if my_table.rows:
            yield my_table
