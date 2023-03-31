__all__ = ['map_docs', 'map_docs_batched']
from contextlib import nullcontext
from math import ceil
from multiprocessing.pool import Pool, ThreadPool
from typing import Callable, Generator, Optional, TypeVar, Union

from rich.progress import track

from docarray import BaseDoc
from docarray.array.abstract_array import AnyDocArray
from docarray.helper import _is_lambda_or_partial_or_local_function

T = TypeVar('T', bound=AnyDocArray)
T_doc = TypeVar('T_doc', bound=BaseDoc)


def map_docs(
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

    ---

    ```python
    from docarray import DocArray
    from docarray.documents import ImageDoc
    from docarray.utils.map import map_docs


    def load_url_to_tensor(img: ImageDoc) -> ImageDoc:
        img.tensor = img.url.load()
        return img


    url = (
        'https://upload.wikimedia.org/wikipedia/commons/8/80/'
        'Dag_Sebastian_Ahlander_at_G%C3%B6teborg_Book_Fair_2012b.jpg'
    )

    da = DocArray[ImageDoc]([ImageDoc(url=url) for _ in range(100)])
    da = DocArray[ImageDoc](
        list(map_docs(da, load_url_to_tensor, backend='thread'))
    )  # threading is usually a good option for IO-bound tasks such as loading an
    # ImageDoc from url

    for doc in da:
        assert doc.tensor is not None
    ```

    ---

    :param da: DocArray to apply function to
    :param func: a function that takes a :class:`BaseDoc` as input and outputs
        a :class:`BaseDoc`.
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

    :return: yield Documents returned from `func`
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


def map_docs_batched(
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
    Each element in the returned iterator is an :class:`AnyDocArray`.

    ---

    ```python
    from docarray import BaseDoc, DocArray
    from docarray.utils.map import map_docs_batched


    class MyDoc(BaseDoc):
        name: str


    def upper_case_name(da: DocArray[MyDoc]) -> DocArray[MyDoc]:
        da.name = [n.upper() for n in da.name]
        return da


    batch_size = 16
    da = DocArray[MyDoc]([MyDoc(name='my orange cat') for _ in range(100)])
    it = map_docs_batched(da, upper_case_name, batch_size=batch_size)
    for i, d in enumerate(it):
        da[i * batch_size : (i + 1) * batch_size] = d

    assert len(da) == 100
    print(da.name[:3])
    ```

    ---

    ```
    ['MY ORANGE CAT', 'MY ORANGE CAT', 'MY ORANGE CAT']
    ```

    ---

    :param da: DocArray to apply function to
    :param batch_size: Size of each generated batch (except the last one, which might
        be smaller).
    :param shuffle: If set, shuffle the Documents before dividing into minibatches.
    :param func: a function that takes an :class:`AnyDocArray` as input and outputs
        an :class:`AnyDocArray` or a :class:`BaseDoc`.
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

    :return: yield DocArrays returned from `func`
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
