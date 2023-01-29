from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement

    from docarray.typing.tensor.abstract_tensor import AbstractTensor


class TensorDisplay:
    """
    Rich representation of a tensor.
    """

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

            tensor_normalized = comp_be.minmax_normalize(
                comp_be.detach(self.tensor), (0, 5)
            )

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

            yield Text(f'{type(self.tensor)} of shape {comp_be.shape(self.tensor)}')

    def __rich_measure__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'Measurement':
        from rich.measure import Measurement

        return Measurement(1, options.max_width)
