from contextlib import nullcontext
from multiprocessing.pool import Pool, ThreadPool
from types import LambdaType
from typing import Any, Callable, Generator, Optional, TypeVar, Union

from docarray import BaseDocument
from docarray.array.abstract_array import AnyDocumentArray

T = TypeVar('T', bound=AnyDocumentArray)
T_Doc = TypeVar('T_Doc', bound=BaseDocument)


def apply(
    da: T,
    func: Callable[[T_Doc], T_Doc],
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> T:
    """
    Apply `func` to every Document of the given DocumentArray while multiprocessing,
    return itself after modification, without in-place changes.

    :param da: DocumentArray to apply function to
    :param func: a function that takes ab:class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`.
    :param backend: `thread` for multi-threading and `process` for multi-processing.
        Defaults to `thread`. In general, if `func` is IO-bound then `thread` is a
        good choice. If `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.

        .. warning::
            When using `process` backend, your `func` should not modify elements in-place.
            This is because the multiprocessing backend passes the variable via pickle
            and works in another process.
            The passed object and the original object do **not** share the same memory.

    :param num_worker: the number of parallel workers. If not given, the number of
        CPUs in the system will be used.
    :param pool: use an existing/external process or thread pool. If given, you will
        be responsible for closing the pool.
    :param show_progress: show a progress bar. Defaults to False.

    :return: DocumentArray with applied modifications
    """
    da_new = da.__class_getitem__(item=da.document_type)()
    for i, doc in enumerate(_map(da, func, backend, num_worker, pool, show_progress)):
        da_new.append(doc)
    return da_new


def _map(
    da: T,
    func: Callable[[T_Doc], T_Doc],
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> Generator[T_Doc, None, None]:
    """
    Return an iterator that applies `func` to every Document in `da` in parallel,
    yielding the results.

    :param da: DocumentArray to apply function to
    :param func:a function that takes ab:class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`. You can either modify elements in-place or return
        new Documents.
    :param backend: `thread` for multi-threading and `process` for multi-processing.
        Defaults to `thread`. In general, if `func` is IO-bound then `thread` is a
        good choice. If `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.

        .. warning::
            When using `process` backend, your `func` should not modify elements in-place.
            This is because the multiprocessing backend passes the variable via pickle
            and works in another process.
            The passed object and the original object do **not** share the same memory.

    :param num_worker: the number of parallel workers. If not given, the number of CPUs
        in the system will be used.
    :param pool: use an existing/external process or thread pool. If given, you will
        be responsible for closing the pool.
    :param show_progress: show a progress bar. Defaults to False.

    :yield: Documents returned from `func`
    """
    from rich.progress import track

    if backend == 'process' and _is_lambda_or_partial_or_local_function(func):
        raise ValueError(
            f'Multiprocessing does not allow functions that are local, lambda or partial: {func}'
        )

    ctx_p: Union[nullcontext, Union[Pool, ThreadPool]]
    if pool:
        p = pool
        ctx_p = nullcontext()
    else:
        p = _get_pool(backend, num_worker)
        ctx_p = p

    with ctx_p:
        imap = p.imap(func, da)
        for x in track(imap, total=len(da), disable=not show_progress):
            yield x


def _get_pool(backend, num_worker) -> Union[Pool, ThreadPool]:
    if backend == 'thread':
        return ThreadPool(processes=num_worker)
    elif backend == 'process':
        return Pool(processes=num_worker)
    else:
        raise ValueError(
            f'`backend` must be either `process` or `thread`, receiving {backend}'
        )


def _is_lambda_or_partial_or_local_function(func: Callable[[Any], Any]) -> bool:
    return (
        (isinstance(func, LambdaType) and func.__name__ == '<lambda>')
        or not hasattr(func, '__qualname__')
        or ('<locals>' in func.__qualname__)
    )
