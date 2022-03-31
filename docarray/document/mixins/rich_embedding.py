import colorsys

from rich.color import Color
from rich.console import Console
from rich.console import ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style

from ...math.helper import minmax_normalize
from ...math.ndarray import to_numpy_array


class ColorBoxEmbedding:
    def __init__(self, array):
        self._array = minmax_normalize(to_numpy_array(array).reshape(-1), (0, 5))

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        h = 0.75
        for idx, y in enumerate(self._array):
            l = 0.1 + ((y / 5) * 0.7)
            r2, g2, b2 = colorsys.hls_to_rgb(h, l + 0.7 / 10, 1.0)
            color = Color.from_rgb(r2 * 255, g2 * 255, b2 * 255)
            yield Segment('▄', Style(color=color, bgcolor=color))
            if idx != 0 and idx % options.max_width == 0:
                yield Segment.line()

    def __rich_measure__(
        self, console: "Console", options: ConsoleOptions
    ) -> Measurement:
        return Measurement(1, options.max_width)


def pixels_to_ascii(image):
    ASCII_CHARS = ["@", "#", "S", "%", "?", "*", "+", ";", "?", ":", "."]
    characters = "".join([ASCII_CHARS[pixel // 25] for pixel in image])
    return characters


class ASCIIEmbedding:
    def __init__(self, array):
        self._array = array

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:

        for idx, y in enumerate(pixels_to_ascii(self._array)):
            yield Segment(y)
            if idx != 0 and idx % options.max_width == 0:
                yield Segment.line()

    def __rich_measure__(
        self, console: "Console", options: ConsoleOptions
    ) -> Measurement:
        return Measurement(1, options.max_width)
