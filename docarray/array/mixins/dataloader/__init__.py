from pathlib import Path
from typing import Union, Optional, Callable, TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from docarray import DocumentArray
    from docarray.typing import T
    from multiprocessing.pool import ThreadPool, Pool


class DataLoaderMixin:
    @classmethod
    def dataloader(
        cls,
        path: Union[str, Path],
        func: Callable[['DocumentArray'], 'T'],
        batch_size: int,
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
        backend: str = 'thread',
        num_worker: Optional[int] = None,
        pool: Optional[Union['Pool', 'ThreadPool']] = None,
        show_progress: bool = False,
    ) -> Generator['DocumentArray', None, None]:
        """Load array elements, batches and maps them with a function in parallel, finally yield the batch in DocumentArray

        :param path: Path or filename where the data is stored.
        :param func: a function that takes :class:`DocumentArray` as input and outputs anything. You can either modify elements
            in-place (only with `thread` backend) or work later on return elements.
        :param batch_size: Size of each generated batch (except the last one, which might be smaller)
        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :param backend: if to use multi-`process` or multi-`thread` as the parallelization backend. In general, if your
            ``func`` is IO-bound then perhaps `thread` is good enough. If your ``func`` is CPU-bound then you may use `process`.
            In practice, you should try yourselves to figure out the best value. However, if you wish to modify the elements
            in-place, regardless of IO/CPU-bound, you should always use `thread` backend.

            .. warning::
                When using `process` backend, you should not expect ``func`` modify elements in-place. This is because
                the multiprocessing backing pass the variable via pickle and work in another process. The passed object
                and the original object do **not** share the same memory.

        :param num_worker: the number of parallel workers. If not given, then the number of CPUs in the system will be used.
        :param pool: use an existing/external pool. If given, `backend` is ignored and you will be responsible for closing the pool.
        :param show_progress: if set, show a progressbar
        :return:
        """
        from .helper import DocumentArrayLoader

        for da in DocumentArrayLoader(
            path, protocol=protocol, compress=compress, show_progress=False
        ).map_batch(
            func,
            batch_size=batch_size,
            backend=backend,
            num_worker=num_worker,
            pool=pool,
            show_progress=show_progress,
        ):
            yield da
