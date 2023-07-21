import importlib
import os
import re
import types
from typing import Any, Optional

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    torch_imported = True


try:
    import tensorflow as tf  # type: ignore # noqa: F401
except (ImportError, TypeError):
    tf_imported = False
else:
    tf_imported = True


try:
    import jax.numpy as jnp  # type: ignore # noqa: F401
except (ImportError, TypeError):
    jnp_imported = False
else:
    jnp_imported = True

INSTALL_INSTRUCTIONS = {
    'google.protobuf': '"docarray[proto]"',
    'lz4': '"docarray[proto]"',
    'pandas': '"docarray[pandas]"',
    'PIL': '"docarray[image]"',
    'pydub': '"docarray[audio]"',
    'av': '"docarray[video]"',
    'trimesh': '"docarray[mesh]"',
    'hnswlib': '"docarray[hnswlib]"',
    'elasticsearch': '"docarray[elasticsearch]"',
    'elastic_transport': '"docarray[elasticsearch]"',
    'weaviate': '"docarray[weaviate]"',
    'qdrant_client': '"docarray[qdrant]"',
    'fastapi': '"docarray[web]"',
    'torch': '"docarray[torch]"',
    'tensorflow': 'protobuf==3.19.0 tensorflow',
    'hubble': '"docarray[jac]"',
    'smart_open': '"docarray[aws]"',
    'boto3': '"docarray[aws]"',
    'botocore': '"docarray[aws]"',
    'redis': '"docarray[redis]"',
    'pymilvus': '"docarray[milvus]"',
}


def import_library(
    package: str, raise_error: bool = True
) -> Optional[types.ModuleType]:
    lib: Optional[types.ModuleType]
    try:
        lib = importlib.import_module(package)
    except (ModuleNotFoundError, ImportError):
        lib = None

    if lib is None and raise_error:
        raise ImportError(
            f'The following required library is not installed: {package} \n'
            f'To install all necessary libraries, run: `pip install {INSTALL_INSTRUCTIONS[package]}`.'
        )
    else:
        return lib


def _get_path_from_docarray_root_level(file_path: str) -> str:
    path = os.path.dirname(file_path)
    rel_path = re.sub('(?s:.*)docarray', 'docarray', path).replace('/', '.')
    return rel_path


def is_torch_available():
    return torch_imported


def is_tf_available():
    return tf_imported


def is_jax_available():
    return jnp_imported


def is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it


def is_notebook() -> bool:
    """
    Check if we're running in a Jupyter notebook, using magic command
    `get_ipython` that only available in Jupyter.

    :return: True if run in a Jupyter notebook else False.
    """

    try:
        shell = get_ipython().__class__.__name__  # type: ignore
    except NameError:
        return False

    if shell == 'ZMQInteractiveShell':
        return True

    elif shell == 'Shell':
        return True

    elif shell == 'TerminalInteractiveShell':
        return False

    else:
        return False
