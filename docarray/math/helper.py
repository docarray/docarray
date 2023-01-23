from typing import Optional, Tuple

import numpy as np


def minmax_normalize(
    x: 'np.ndarray',
    t_range: Tuple = (0, 1),
    x_range: Optional[Tuple] = None,
    eps: float = 1e-7,
):
    """Normalize values in `x` into `t_range`.

    `x` can be a 1D array or a 2D array. When `x` is a 2D array, then normalization is
    row-based.

    .. note::
        - with `t_range=(0, 1)` will normalize the min-value of the data to 0, max to 1;
        - with `t_range=(1, 0)` will normalize the min-value of the data to 1, max value
          of the data to 0.

    :param x: the data to be normalized
    :param t_range: a tuple represents the target range.
    :param x_range: a tuple represents x range.
    :param eps: a small jitter to avoid divde by zero
    :return: normalized data in `t_range`
    """
    a, b = t_range

    min_d = x_range[0] if x_range else np.min(x, axis=-1, keepdims=True)
    max_d = x_range[1] if x_range else np.max(x, axis=-1, keepdims=True)
    r = (b - a) * (x - min_d) / (max_d - min_d + eps) + a

    return np.clip(r, *((a, b) if a < b else (b, a)))
