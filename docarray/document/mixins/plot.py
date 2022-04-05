from ...helper import deprecate_by


class PlotMixin:
    """Provide helper functions for :class:`Document` to plot and visualize itself."""

    def _ipython_display_(self):
        """Displays the object in IPython as a side effect"""
        self.summary()

    def __rich_console__(self, console, options):

        yield f":page_facing_up: [b]Document[/b]: [cyan]{self.id}[cyan]"
        from rich.table import Table
        from rich import box

        my_table = Table(
            'Attribute', 'Value', width=80, box=box.ROUNDED, highlight=True
        )
        for f in self.non_empty_fields:
            if f in ('embedding', 'tensor'):
                from ...math.ndarray import to_numpy_array

                v = to_numpy_array(getattr(self, f))
                if v.squeeze().ndim == 1:
                    from .rich_embedding import ColorBoxEmbedding

                    v = ColorBoxEmbedding(v.squeeze())
                else:
                    v = str(getattr(self, f))
                my_table.add_row(f, v)
            elif f not in ('id', 'chunks', 'matches'):
                my_table.add_row(f, str(getattr(self, f)))
        if my_table.rows:
            yield my_table

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from rich import print

        print(self._plot_recursion())

    def _plot_recursion(self, tree=None):
        if tree is None:
            from rich.tree import Tree

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

    def display(self):
        """Plot image data from :attr:`.tensor` or :attr:`.uri`."""
        from IPython.display import Image, display

        if self.uri:
            if self.mime_type.startswith('audio') or self.uri.startswith('data:audio/'):
                uri = _convert_display_uri(self.uri, self.mime_type)
                _html5_audio_player(uri)
            elif self.mime_type.startswith('video') or self.uri.startswith(
                'data:video/'
            ):
                uri = _convert_display_uri(self.uri, self.mime_type)
                _html5_video_player(uri)
            elif self.uri.startswith('data:image/'):
                _html5_image(self.uri)
            else:
                display(Image(self.uri))
        elif self.tensor is not None:
            try:
                import PIL.Image

                p = PIL.Image.fromarray(self.tensor)
                if p.mode != 'RGB':
                    raise
                display(p)
            except:
                import matplotlib.pyplot as plt

                plt.matshow(self.tensor)
        else:
            self.summary()

    plot = deprecate_by(display, removed_at='0.5')


def _convert_display_uri(uri, mime_type):
    import urllib
    from .helper import _to_datauri, _uri_to_blob

    scheme = urllib.parse.urlparse(uri).scheme

    if scheme not in ['data', 'http', 'https']:
        blob = _uri_to_blob(uri)
        return _to_datauri(mime_type, blob)
    return uri


def _html5_image(uri):
    from IPython.display import display
    from IPython.core.display import HTML  # noqa

    src = f'''
    <body>
    <image src="{uri}" height="200px">
    </body>
    '''
    display(HTML(src))  # noqa


def _html5_video_player(uri):
    from IPython.display import display
    from IPython.core.display import HTML  # noqa

    src = f'''
    <body>
    <video width="320" height="240" autoplay muted controls>
    <source src="{uri}">
    Your browser does not support the video tag.
    </video>
    </body>
    '''
    display(HTML(src))  # noqa


def _html5_audio_player(uri):
    from IPython.display import display
    from IPython.core.display import HTML  # noqa

    src = f'''
    <body>
    <audio controls="controls" style="width:320px" >
      <source src="{uri}"/>
      Your browser does not support the audio element.
    </audio>
    </body>
    '''
    display(HTML(src))  # noqa
