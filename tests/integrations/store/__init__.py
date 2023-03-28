import tracemalloc
from functools import wraps

from docarray import DocArray
from docarray.documents import TextDoc


def get_test_da(n: int):
    return DocArray[TextDoc](gen_text_docs(n))


def gen_text_docs(n: int):
    for i in range(n):
        yield TextDoc(text=f'text {i}')


def profile_memory(func):
    """Decorator to profile memory usage of a function.

    Returns:
        original function return value, (current memory usage, peak memory usage)
    """

    @wraps(func)
    def _inner(*args, **kwargs):
        tracemalloc.start()
        ret = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ret, (current, peak)

    return _inner
