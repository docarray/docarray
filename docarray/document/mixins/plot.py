import copy
from typing import Optional

import numpy as np

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
            if f.startswith('_'):
                continue
            elif f in ('text', 'blob') and len(getattr(self, f)) > 100:
                v = getattr(self, f)
                my_table.add_row(f, str(v)[:100] + f'... [dim](length: {len(v)})')
            elif f in ('embedding', 'tensor'):
                from ...math.ndarray import to_numpy_array

                v = to_numpy_array(getattr(self, f))
                if v.squeeze().ndim == 1 and len(v) < 1000:
                    from .rich_embedding import ColorBoxEmbedding

                    v = ColorBoxEmbedding(v.squeeze())
                else:
                    v = f'{type(getattr(self, f))} in shape {v.shape}, dtype: {v.dtype}'
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

    def plot_matches_sprites(
        self,
        top_k: int = 10,
        channel_axis: int = -1,
        inv_normalize: bool = False,
        skip_empty: bool = False,
        canvas_size: int = 1920,
        min_size: int = 100,
        output: Optional[str] = None,
    ):
        """Generate a sprite image for the query and its matching images in this Document object.

        An image sprite is a collection of images put into a single image. Query image is on the left
        followed by matching images. The Document object should contain matches.

        :param top_k: the number of top matching documents to show in the sprite.
        :param channel_axis: the axis id of the color channel, ``-1`` indicates the color channel info at the last axis
        :param inv_normalize: If set to True, inverse the normalization of a float32 image :attr:`.tensor` into a uint8
            image :attr:`.tensor` inplace.
        :param skip_empty: skip matches which has no .uri or .tensor.
        :param canvas_size: the width of the canvas
        :param min_size: the minimum size of the image
        :param output: Optional path to store the visualization. If not given, show in UI
        """
        if not self or not self.matches:
            raise ValueError(f'{self!r} is empty or has no matches')

        if not self.uri and self.tensor is None:
            raise ValueError(
                f'Document has neither `uri` nor `tensor`, cannot be plotted'
            )

        if top_k <= 0:
            raise ValueError(f'`limit` must be larger than 0, receiving {top_k}')

        import matplotlib.pyplot as plt

        img_per_row = top_k + 2
        if top_k > len(self.matches):
            img_per_row = len(self.matches) + 2

        img_size = int((canvas_size - 50) / img_per_row)
        if img_size < min_size:
            # image is too small, recompute the image size and canvas size
            img_size = min_size
            canvas_size = img_per_row * img_size + 50

        try:
            _d = copy.deepcopy(self)
            if _d.content_type != 'tensor':
                _d.load_uri_to_image_tensor()  # the channel axis is -1

            if inv_normalize:
                # inverse normalise to uint8 and set the channel axis to -1
                _d.set_image_tensor_inv_normalization(channel_axis)

            _d.set_image_tensor_channel_axis(channel_axis, -1)

            # Maintain the aspect ratio keeping the width fixed
            h, w, _ = _d.tensor.shape
            img_h, img_w = int(h * (img_size / float(w))), img_size

            sprite_img = np.ones([img_h + 20, canvas_size, 3], dtype='uint8')

            _d.set_image_tensor_shape(shape=(img_h, img_w))

            sprite_img[10 : img_h + 10, 10 : 10 + img_w] = _d.tensor
            pos = canvas_size // img_per_row

            for col_id, d in enumerate(self.matches, start=2):
                if not d.uri and d.tensor is None:
                    if skip_empty:
                        continue
                    else:
                        raise ValueError(
                            f'Document match has neither `uri` nor `tensor`, cannot be plotted'
                        )
                _d = copy.deepcopy(d)
                if _d.content_type != 'tensor':
                    _d.load_uri_to_image_tensor()

                if inv_normalize:
                    _d.set_image_tensor_inv_normalization(channel_axis=channel_axis)

                _d.set_image_tensor_channel_axis(
                    channel_axis, -1
                ).set_image_tensor_shape(shape=(img_h, img_w))

                # paste it on the main canvas
                sprite_img[
                    10 : img_h + 10,
                    (col_id * pos) : ((col_id * pos) + img_w),
                ] = _d.tensor

                col_id += 1
                if col_id >= img_per_row:
                    break
        except Exception as ex:
            raise ValueError('Bad image tensor. Try different `channel_axis`') from ex
        from PIL import Image

        im = Image.fromarray(sprite_img)

        if output:
            with open(output, 'wb') as fp:
                im.save(fp)
        else:
            plt.figure(figsize=(img_per_row, 2))
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(im, interpolation="none")
            plt.show()


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
