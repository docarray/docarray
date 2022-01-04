import base64
from typing import Optional

from ...helper import random_identity, download_mermaid_url


class PlotMixin:
    """Provide helper functions for :class:`Document` to plot and visualize itself. """

    def _ipython_display_(self):
        """Displays the object in IPython as a side effect"""
        self.summary()

    def summary(self) -> None:
        """ Print non-empty fields and nested structure of this Document object."""
        _str_list = []
        self._plot_recursion(_str_list, indent=0)
        print('\n'.join(_str_list))

    def _plot_recursion(self, _str_list, indent, box_char='├─'):
        prefix = (' ' * indent + box_char) if indent else ''
        _str_list.append(f'{prefix} {self}')

        for a in ('matches', 'chunks'):
            if getattr(self, a):
                prefix = ' ' * (indent + 4) + '└─'
                _str_list.append(f'{prefix} {a}')

                for d in getattr(self, a)[:-1]:
                    d._plot_recursion(_str_list, indent=len(prefix) + 4)
                getattr(self, a)[-1]._plot_recursion(
                    _str_list, indent=len(prefix) + 4, box_char='└─'
                )

    def plot(self):
        """ Plot image data from :attr:`.blob` or :attr:`.uri`. """
        from IPython.display import Image, display

        if self.blob is not None:
            import PIL.Image

            display(PIL.Image.fromarray(self.blob))
        elif self.uri:
            display(Image(self.uri))
        else:
            raise ValueError('`uri` and `blob` is empty')
