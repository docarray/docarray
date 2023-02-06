try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    torch_imported = True


def is_torch_available():
    return torch_imported
