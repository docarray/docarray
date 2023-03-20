import os
from pathlib import Path


def get_cache_path() -> Path:
    """
    Get the path to the cache directory.

    :return: The path to the cache directory.
    """
    cache_path = Path.home() / '.cache' / 'docarray'
    if "DOCARRAY_CACHE" in os.environ:
        cache_path = Path(os.environ["DOCARRAY_CACHE"])
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
