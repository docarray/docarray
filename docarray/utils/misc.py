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


def is_torch_available():
    return torch_imported


def is_tf_available():
    return tf_imported


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
