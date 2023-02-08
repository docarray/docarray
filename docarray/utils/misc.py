try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    torch_imported = True


try:
    import tensorflow as tf  # noqa: F401
except (ImportError, TypeError):
    tf_imported = False
else:
    tf_imported = True


def is_torch_available():
    return torch_imported


def is_tf_available():
    return tf_imported
