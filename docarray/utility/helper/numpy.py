import warnings
from typing import Optional, Tuple

import numpy as np


def top_k(
    values: 'np.ndarray',
    k: int,
    descending: bool = False,
    device: Optional[str] = None,
) -> Tuple['np.ndarray', 'np.ndarray']:
    """Retrieves the top k smallest values in `values`,
    and returns them alongside their indices in the input `values`.
    Can also be used to retrieve the top k largest values,
    by setting the `descending` flag.

    :param values: Torch tensor of values to rank.
        Should be a vector (single axis), or squeezable.
        TODO(johannes) is squeezable a term understood in np land?
    :param k: number of values to retrieve
    :param descending: retrieve largest values instead of smallest values
    :param device: Not supported for this backend
    :return: Tuple containing the retrieved values, and their indices
    """
    if device is not None:
        warnings.warn('`device` is not supported for numpy operations')

    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=0)

    if descending:
        values = -values

    if k >= values.shape[1]:
        idx = values.argsort(axis=1)[:, :k]
        values = np.take_along_axis(values, idx, axis=1)
    else:
        idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
        values = np.take_along_axis(values, idx_ps, axis=1)
        idx_fs = values.argsort(axis=1)
        idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
        values = np.take_along_axis(values, idx_fs, axis=1)

    if descending:
        values = -values

    return values.squeeze(), idx.squeeze()
