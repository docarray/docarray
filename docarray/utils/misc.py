from typing import Any

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    torch_imported = True


def is_torch_available():
    return torch_imported


def is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it
