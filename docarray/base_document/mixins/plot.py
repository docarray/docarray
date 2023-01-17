from typing import Optional

from rich.tree import Tree


class PlotMixin:
    def display(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from rich import print

        print(self._plot_recursion())

    def _plot_recursion(self, tree: Optional[Tree] = None):
        if tree is None:

            tree = Tree(self)
        else:
            tree = tree.add(self)
        for a in ('matches', 'chunks'):
            if getattr(self, a):
                if a == 'chunks':
                    _icon = ':diamond_with_a_dot:'
                else:
                    _icon = ':large_orange_diamond:'
                _match_tree = tree.add(f'{_icon} [b]{a.capitalize()}[/b]')
                for d in getattr(self, a):
                    d._plot_recursion(_match_tree)
        return tree
