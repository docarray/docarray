from contextlib import nullcontext
from math import ceil
from multiprocessing.pool import Pool, ThreadPool
from typing import Callable, Generator, Optional, TypeVar, Union

from rich.progress import track

from docarray import BaseDocument
from docarray.array.abstract_array import AnyDocumentArray
from docarray.helper import _is_lambda_or_partial_or_local_function

T = TypeVar('T', bound=AnyDocumentArray)
T_doc = TypeVar('T_doc', bound=BaseDocument)


def apply(
    da: T,
    func: Callable[[T_doc], T_doc],
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> None:
    """
    Apply `func` to every Document of the given DocumentArray in-place while multithreading
    or multiprocessing.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocumentArray
        from docarray.documents import Image
        from docarray.utils.apply import apply


        def load_url_to_tensor(img: Image) -> Image:
            img.tensor = img.url.load()
            return img


        da = DocumentArray[Image]([Image(url='path/to/img.png') for _ in range(100)])
        apply(
            da, load_url_to_tensor, backend='thread'
        )  # threading is usually a good option for IO-bound tasks such as loading an image from url

        for doc in da:
            assert doc.tensor is not None

    :param da: DocumentArray to apply function to
    :param func: a function that takes a :class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`.
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
        In general, if `func` is IO-bound then `thread` is a good choice.
        On the other hand, if `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.
        Note that computation that is offloaded to non-python code (e.g. through np/torch/tf)
        falls under the "IO-bound" category.

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

    """
    for i, doc in enumerate(_map(da, func, backend, num_worker, pool, show_progress)):
        da[i] = doc


def _map(
    da: T,
    func: Callable[[T_doc], T_doc],
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> Generator[T_doc, None, None]:
    """
    Return an iterator that applies `func` to every Document in `da` in parallel,
    yielding the results.

    .. seealso::
        - To return :class:`DocumentArray`, please use :func:`apply`.

    :param da: DocumentArray to apply function to
    :param func: a function that takes a :class:`BaseDocument` as input and outputs
        a :class:`BaseDocument`.
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
        In general, if `func` is IO-bound then `thread` is a good choice.
        On the other hand, if `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.
        Note that computation that is offloaded to non-python code (e.g. through np/torch/tf)
        falls under the "IO-bound" category.

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

    if backend == 'process' and _is_lambda_or_partial_or_local_function(func):
        raise ValueError(
            f'Multiprocessing does not allow functions that are local, lambda or partial: {func}'
        )

    context_pool: Union[nullcontext, Union[Pool, ThreadPool]]
    if pool:
        p = pool
        context_pool = nullcontext()
    else:
        p = _get_pool(backend, num_worker)
        context_pool = p

    with context_pool:
        imap = p.imap(func, da)
        for x in track(imap, total=len(da), disable=not show_progress):
            yield x


def apply_batch(
    da: T,
    func: Callable[[T], Union[T, T_doc]],
    batch_size: int,
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    shuffle: bool = False,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> None:
    """
    Batches itself into mini-batches, applies `func` to every mini-batch in-place.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import BaseDocument, DocumentArray
        from docarray.utils.apply import apply_batch


        class MyDoc(BaseDocument):
            name: str


        def upper_case_name(da: DocumentArray[MyDoc]) -> DocumentArray[MyDoc]:
            da.name = [n.upper() for n in da.name]
            return da


        da = DocumentArray[MyDoc]([MyDoc(name='my orange cat') for _ in range(100)])
        apply_batch(da, upper_case_name, batch_size=10)
        print(da.name[:3])

    .. code-block:: text

        ['MY ORANGE CAT', 'MY ORANGE CAT', 'MY ORANGE CAT']

    :param da: DocumentArray to apply function to
    :param func: a function that takes an :class:`AnyDocumentArray` as input and outputs
        an :class:`AnyDocumentArray` or a :class:`BaseDocument`.
    :param batch_size: size of each generated batch (except the last batch, which might
        be smaller).
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
        In general, if `func` is IO-bound then `thread` is a good choice.
        On the other hand, if `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.
        Note that computation that is offloaded to non-python code (e.g. through np/torch/tf)
        falls under the "IO-bound" category.

        .. warning::
            When using `process` backend, your `func` should not modify elements in-place.
            This is because the multiprocessing backend passes the variable via pickle
            and works in another process.
            The passed object and the original object do **not** share the same memory.

    :param num_worker: the number of parallel workers. If not given, the number of CPUs
        in the system will be used.
    :param shuffle: If set, shuffle the Documents before dividing into minibatches.
    :param pool: use an existing/external process or thread pool. If given, you will
        be responsible for closing the pool.
    :param show_progress: show a progress bar. Defaults to False.
    """
    for i, batch in enumerate(
        _map_batch(
            da, func, batch_size, backend, num_worker, shuffle, pool, show_progress
        )
    ):
        start = i * batch_size
        stop = (i + 1) * batch_size
        if isinstance(batch, BaseDocument):
            da[start:stop] = da.__class_getitem__(da.document_type)([batch])
        else:
            da[start:stop] = batch


def _map_batch(
    da: T,
    func: Callable[[T], Union[T, T_doc]],
    batch_size: int,
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    shuffle: bool = False,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> Generator[Union[T, T_doc], None, None]:
    """
    Return an iterator that applies `func` to every **minibatch** of iterable in parallel,
    yielding the results.
    Each element in the returned iterator is an :class:`AnyDocumentArray`.

    .. seealso::
        - To return :class:`DocumentArray`, please use :func:`apply_batch`.

    :param batch_size: Size of each generated batch (except the last one, which might
        be smaller).
    :param shuffle: If set, shuffle the Documents before dividing into minibatches.
    :param func: a function that takes an :class:`AnyDocumentArray` as input and outputs
        an :class:`AnyDocumentArray` or a :class:`BaseDocument`.
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
        In general, if `func` is IO-bound then `thread` is a good choice.
        On the other hand, if `func` is CPU-bound, then you may use `process`.
        In practice, you should try yourselves to figure out the best value.
        However, if you wish to modify the elements in-place, regardless of IO/CPU-bound,
        you should always use `thread` backend.
        Note that computation that is offloaded to non-python code (e.g. through np/torch/tf)
        falls under the "IO-bound" category.

        .. warning::
            When using `process` backend, your `func` should not modify elements in-place.
            This is because the multiprocessing backend passes the variable via pickle
            and works in another process.
            The passed object and the original object do **not** share the same memory.

    :param num_worker: the number of parallel workers. If not given, then the number of CPUs
        in the system will be used.
    :param show_progress: show a progress bar
    :param pool: use an existing/external pool. If given, `backend` is ignored and you will
        be responsible for closing the pool.

    :yield: DocumentArrays returned from `func`
    """
    if backend == 'process' and _is_lambda_or_partial_or_local_function(func):
        raise ValueError(
            f'Multiprocessing does not allow functions that are local, lambda or partial: {func}'
        )

    context_pool: Union[nullcontext, Union[Pool, ThreadPool]]
    if pool:
        p = pool
        context_pool = nullcontext()
    else:
        p = _get_pool(backend, num_worker)
        context_pool = p

    with context_pool:
        imap = p.imap(func, da._batch(batch_size=batch_size, shuffle=shuffle))
        for x in track(
            imap, total=ceil(len(da) / batch_size), disable=not show_progress
        ):
            yield x


def _get_pool(backend, num_worker) -> Union[Pool, ThreadPool]:
    """
    Get Pool instance for multiprocessing or ThreadPool instance for multithreading.
    """
    if backend == 'thread':
        return ThreadPool(processes=num_worker)
    elif backend == 'process':
        return Pool(processes=num_worker)
    else:
        raise ValueError(
            f'`backend` must be either `process` or `thread`, receiving {backend}'
        )
