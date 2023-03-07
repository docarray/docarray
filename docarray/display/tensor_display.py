from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement

    from docarray.typing.tensor.abstract_tensor import AbstractTensor


class TensorDisplay:
    """
    Rich representation of a tensor.
    """

    tensor_min_width: int = 30

    def __init__(self, tensor: 'AbstractTensor'):
        self.tensor = tensor

    def __rich_console__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'RenderResult':
        comp_be = self.tensor.get_comp_backend()
        t_squeezed = comp_be.squeeze(comp_be.detach(self.tensor))

        if comp_be.n_dim(t_squeezed) == 1 and comp_be.shape(t_squeezed)[0] < 200:
            import colorsys

            from rich.color import Color
            from rich.segment import Segment
            from rich.style import Style

            tensor_normalized = comp_be.minmax_normalize(t_squeezed, (0, 5))

            hue = 0.75
            saturation = 1.0
            for idx, y in enumerate(tensor_normalized):
                luminance = 0.1 + ((y / 5) * 0.7)
                r, g, b = colorsys.hls_to_rgb(hue, luminance + 0.07, saturation)
                color = Color.from_rgb(r * 255, g * 255, b * 255)
                yield Segment('▄', Style(color=color, bgcolor=color))
                if idx != 0 and idx % options.max_width == 0:
                    yield Segment.line()
        else:
            from rich.text import Text

            yield Text(
                f'{self.tensor.__class__.__name__} of '
                f'shape {comp_be.shape(self.tensor)}, '
                f'dtype: {str(comp_be.dtype(self.tensor))}'
            )

    def __rich_measure__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'Measurement':
        from rich.measure import Measurement

        width = self._compute_table_width(max_width=options.max_width)
        return Measurement(1, width)

    def _compute_table_width(self, max_width: int) -> int:
        """
        Compute the width of the table. Depending on the length of the tensor, the width
        should be in the range of 30 (min) and a given `max_width`.
        :return: the width of the table
        """
        comp_be = self.tensor.get_comp_backend()
        t_squeezed = comp_be.squeeze(comp_be.detach(self.tensor))
        if comp_be.n_dim(t_squeezed) == 1 and comp_be.shape(t_squeezed)[0] < max_width:
            min_capped = max(comp_be.shape(t_squeezed)[0], self.tensor_min_width)
            min_max_capped = min(min_capped, max_width)
            return min_max_capped
        else:
            return max_width
