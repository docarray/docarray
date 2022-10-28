import sys
from contextlib import nullcontext
from math import ceil
from types import LambdaType
from typing import (
    Callable,
    TYPE_CHECKING,
    Generator,
    Optional,
    overload,
    TypeVar,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import T
    from docarray import Document, DocumentArray
    from multiprocessing.pool import ThreadPool, Pool


T_DA = TypeVar('T_DA')


class ParallelMixin:
    """Helper functions that provide parallel map to :class:`DocumentArray`"""

    # overload_inject_start_apply
    @overload
    def apply(
        self: 'T',
        func: Callable[['Document'], 'Document'],
        backend: str = 'thread',
        num_worker: Optional[int] = None,
        show_progress: bool = False,
        pool: Optional[Union['Pool', 'ThreadPool']] = None,
    ) -> 'T':
        """Apply ``func`` to every Document in itself, return itself after modification.

        :param func: a function that takes :class:`Document` as input and outputs :class:`Document`.
        :param backend: `thread` for multi-threading and `process` for multi-processing. Defaults to `thread`.
            In general, if your
            ``func`` is IO-bound then `thread` is a good choice. If your ``func`` is CPU-bound, then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backend passes the variable via pickle and works in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param pool: use an existing/external process or thread pool. If given, `backend` is ignored and you will be responsible for closing the pool.
        :param show_progress: show a progress bar

        """

    # overload_inject_end_apply

    def apply(self: 'T', *args, **kwargs) -> 'T':
        # implementation_stub_inject_start_apply

        """Apply ``func`` to every Document in itself, return itself after modification.

        :param func: a function that takes :class:`Document` as input and outputs :class:`Document`.
        :param backend: `thread` for multi-threading and `process` for multi-processing. Defaults to `thread`.
            In general, if your
            ``func`` is IO-bound then `thread` is a good choice. If your ``func`` is CPU-bound, then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backend passes the variable via pickle and works in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param pool: use an existing/external process or thread pool. If given, `backend` is ignored and you will be responsible for closing the pool.
        :param show_progress: show a progress bar
        :return: itself after modification

        .. # noqa: DAR102
        .. # noqa: DAR202
        .. # noqa: DAR101
        .. # noqa: DAR003
        """
        # implementation_stub_inject_end_apply
        for doc in self.map(*args, **kwargs):
            self[doc.id] = doc
        return self

    def map(
        self,
        func: Callable[['Document'], 'T'],
        backend: str = 'thread',
        num_worker: Optional[int] = None,
        show_progress: bool = False,
        pool: Optional[Union['Pool', 'ThreadPool']] = None,
    ) -> Generator['T', None, None]:
        """Return an iterator that applies function to every **element** of iterable in parallel, yielding the results.

        .. seealso::
            - To process on a batch of elements, please use :meth:`.map_batch`;
            - To return a :class:`DocumentArray`, please use :meth:`.apply`.

        :param func: a function that takes :class:`Document` as input and outputs anything. You can either modify elements
            in-place (only with `thread` backend) or work later on return elements.
        :param backend: `thread` for multi-threading and `process` for multi-processing. Defaults to `thread`.
            In general, if your
            ``func`` is IO-bound then `thread` is a good choice. If your ``func`` is CPU-bound, then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backing passes the variable via pickle and works in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param show_progress: show a progress bar
        :param pool: use an existing/external process or thread pool. If given, `backend` is ignored and you will be responsible for closing the pool.

        :yield: anything return from ``func``
        """
        if _is_lambda_or_partial_or_local_function(func) and backend == 'process':
            func = _globalize_lambda_function(func)

        from rich.progress import track

        if pool:
            p = pool
            ctx_p = nullcontext()
        else:
            p = _get_pool(backend, num_worker)
            ctx_p = p

        with ctx_p:
            for x in track(
                p.imap(func, self), total=len(self), disable=not show_progress
            ):
                yield x

    # overload_inject_start_apply_batch
    @overload
    def apply_batch(
        self: 'T',
        func: Callable[['DocumentArray'], 'DocumentArray'],
        batch_size: int,
        backend: str = 'thread',
        num_worker: Optional[int] = None,
        shuffle: bool = False,
        show_progress: bool = False,
        pool: Optional[Union['Pool', 'ThreadPool']] = None,
    ) -> 'T':
        """Batches itself into mini-batches, applies `func` to every mini-batch, and return itself after the modifications.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document, DocumentArray

            da = DocumentArray([Document(text='The cake is a lie') for _ in range(100)])


            def func(doc):
                da.texts = [t.upper() for t in da.texts]
                return da


            da.apply_batch(func, batch_size=10)
            print(da.texts[:3])

        .. code-block:: text

            ['THE CAKE IS A LIE', 'THE CAKE IS A LIE', 'THE CAKE IS A LIE']

        :param func: a function that takes :class:`DocumentArray` as input and outputs :class:`DocumentArray`.
        :param backend: `thread` for multi-threading and `process` for multi-processing. Defaults to `thread`.
            In general, if your
            ``func`` is IO-bound then `thread` is a good choice. If your ``func`` is CPU-bound, then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backend passes the variable via pickle and works in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param batch_size: Size of each generated batch (except the last batch, which might be smaller). Default: 32
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :param show_progress: show a progress bar
        :param pool: use an existing/external process or thread pool. If given, `backend` is ignored and you will be responsible for closing the pool.

        """

    # overload_inject_end_apply_batch

    def apply_batch(self: 'T', *args, **kwargs) -> 'T':
        # implementation_stub_inject_start_apply_batch

        """Batches itself into mini-batches, applies `func` to every mini-batch, and return itself after the modifications.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document, DocumentArray

            da = DocumentArray([Document(text='The cake is a lie') for _ in range(100)])


            def func(doc):
                da.texts = [t.upper() for t in da.texts]
                return da


            da.apply_batch(func, batch_size=10)
            print(da.texts[:3])

        .. code-block:: text

            ['THE CAKE IS A LIE', 'THE CAKE IS A LIE', 'THE CAKE IS A LIE']

        :param func: a function that takes :class:`DocumentArray` as input and outputs :class:`DocumentArray`.
        :param backend: `thread` for multi-threading and `process` for multi-processing. Defaults to `thread`.
            In general, if your
            ``func`` is IO-bound then `thread` is a good choice. If your ``func`` is CPU-bound, then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backend passes the variable via pickle and works in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param batch_size: Size of each generated batch (except the last batch, which might be smaller). Default: 32
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :param show_progress: show a progress bar
        :param pool: use an existing/external process or thread pool. If given, `backend` is ignored and you will be responsible for closing the pool.
        :return: itself after modification

        .. # noqa: DAR102
        .. # noqa: DAR202
        .. # noqa: DAR101
        .. # noqa: DAR003
        """
        # implementation_stub_inject_end_apply_batch
        for _b in self.map_batch(*args, **kwargs):
            self[[doc.id for doc in _b]] = _b
        return self

    def map_batch(
        self: 'T_DA',
        func: Callable[['DocumentArray'], 'T'],
        batch_size: int,
        backend: str = 'thread',
        num_worker: Optional[int] = None,
        shuffle: bool = False,
        show_progress: bool = False,
        pool: Optional[Union['Pool', 'ThreadPool']] = None,
    ) -> Generator['T', None, None]:
        """Return an iterator that applies function to every **minibatch** of iterable in parallel, yielding the results.
        Each element in the returned iterator is :class:`DocumentArray`.

        .. seealso::
            - To process single element, please use :meth:`.map`;
            - To return :class:`DocumentArray`, please use :meth:`.apply_batch`.

        :param batch_size: Size of each generated batch (except the last one, which might be smaller, default: 32)
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :param func: a function that takes :class:`DocumentArray` as input and outputs anything. You can either modify elements
            in-place (only with `thread` backend) or work later on return elements.
        :param backend: if to use multi-`process` or multi-`thread` as the parallelization backend. In general, if your
            ``func`` is IO-bound then perhaps `thread` is good enough. If your ``func`` is CPU-bound then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backing pass the variable via pickle and work in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param show_progress: show a progress bar
        :param pool: use an existing/external pool. If given, `backend` is ignored and you will be responsible for closing the pool.

        :yield: anything return from ``func``
        """

        if _is_lambda_or_partial_or_local_function(func) and backend == 'process':
            func = _globalize_lambda_function(func)

        from rich.progress import track

        if pool:
            p = pool
            ctx_p = nullcontext()
        else:
            p = _get_pool(backend, num_worker)
            ctx_p = p

        with ctx_p:
            for x in track(
                p.imap(func, self.batch(batch_size=batch_size, shuffle=shuffle)),
                total=ceil(len(self) / batch_size),
                disable=not show_progress,
            ):
                yield x


def _get_pool(backend, num_worker):
    if backend == 'thread':
        from multiprocessing.pool import ThreadPool as Pool

        return Pool(processes=num_worker)
    elif backend == 'process':
        from multiprocessing.pool import Pool

        return Pool(processes=num_worker)
    else:
        raise ValueError(
            f'`backend` must be either `process` or `thread`, receiving {backend}'
        )


def _is_lambda_or_partial_or_local_function(func):
    return (
        (isinstance(func, LambdaType) and func.__name__ == '<lambda>')
        or not hasattr(func, '__qualname__')
        or ('<locals>' in func.__qualname__)
    )


def _globalize_lambda_function(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    from docarray.helper import random_identity

    result.__name__ = result.__qualname__ = random_identity()
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result
