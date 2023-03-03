from contextlib import nullcontext
from types import LambdaType
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, TypeVar, Union

from docarray import BaseDocument
from docarray.array.abstract_array import AnyDocumentArray

if TYPE_CHECKING:
    from multiprocessing.pool import Pool


T = TypeVar('T', bound=AnyDocumentArray)


def apply(
    da: T,
    func: Callable[[BaseDocument], BaseDocument],
    num_worker: Optional[int] = None,
    pool: Optional['Pool'] = None,
    show_progress: bool = False,
) -> T:
    """
    Apply `func` to every Document of the given DocumentArray while multiprocessing,
    return itself after modification, without in-place changes.

    :param da: DocumentArray to apply function to
    :param func: a function that takes ab:class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`.
    :param num_worker: the number of parallel workers. If not given, the number of
        CPUs in the system will be used.
    :param pool: use an existing/external process or thread pool. If given, you will
        be responsible for closing the pool.
    :param show_progress: show a progress bar. Defaults to False.

    :return: DocumentArray with applied modifications
    """
    da_new = da.__class_getitem__(item=da.document_type)()
    for i, doc in enumerate(_map(da, func, num_worker, pool, show_progress)):
        da_new.append(doc)
    return da_new


def _map(
    da: T,
    func: Callable[[BaseDocument], BaseDocument],
    num_worker: Optional[int] = None,
    pool: Optional['Pool'] = None,
    show_progress: bool = False,
) -> Generator['BaseDocument', None, None]:
    """
    Return an iterator that applies `func` to every Document in `da` in parallel,
    yielding the results.

    :param da: DocumentArray to apply function to
    :param func:a function that takes ab:class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`. You can either modify elements in-place or return
        new Documents.
    :param num_worker: the number of parallel workers. If not given, the number of
        CPUs in the system will be used.
    use an existing/external process or thread pool. If given, you will
        be responsible for closing the pool.
    :param show_progress: show a progress bar. Defaults to False.

    :yield: Documents returned from `func`
    """
    from rich.progress import track

    if _is_lambda_or_partial_or_local_function(func):
        raise ValueError(
            f'Multiprocessing does not allow functions that are local, lambda or partial: {func}'
        )

    ctx_p: Union[nullcontext, 'Pool']
    if pool:
        p = pool
        ctx_p = nullcontext()
    else:
        from multiprocessing.pool import Pool

        p = Pool(processes=num_worker)
        ctx_p = p

    with ctx_p:
        for x in track(p.imap(func, da), total=len(da), disable=not show_progress):
            yield x


def _is_lambda_or_partial_or_local_function(func: Callable[[Any], Any]) -> bool:
    return (
        (isinstance(func, LambdaType) and func.__name__ == '<lambda>')
        or not hasattr(func, '__qualname__')
        or ('<locals>' in func.__qualname__)
    )
