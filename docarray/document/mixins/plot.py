from ...helper import deprecate_by


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

    def display(self):
        """ Plot image data from :attr:`.tensor` or :attr:`.uri`. """
        from IPython.display import Image, display

        if self.uri:
            if self.mime_type.startswith('audio'):
                _html5_audio_player(self.uri)
            elif self.mime_type.startswith('video'):
                _html5_video_player(self.uri)
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


def _html5_video_player(uri):
    from IPython.core.display import HTML  # noqa

    src = f'''
    <body>
    <video width="320" height="240" autoplay muted controls>
    <source src="files/{uri}">
    Your browser does not support the video tag.
    </video>
    </body>
    '''
    display(HTML(src))  # noqa


def _html5_audio_player(uri):
    from IPython.core.display import HTML  # noqa

    src = f'''
    <body>
    <audio controls="controls" style="width:320px" >
      <source src="files/{uri}"/>
      Your browser does not support the audio element.
    </audio>
    </body>
    '''
    display(HTML(src))  # noqa
