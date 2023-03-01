from multiprocessing import cpu_count
from time import time

import pytest

from docarray import BaseDocument, DocumentArray
from docarray.utils.apply import apply


class MyDoc(BaseDocument):
    title: str


def title_da(doc: MyDoc) -> MyDoc:
    return MyDoc(title=doc.title)


@pytest.mark.benchmark
def test_apply_benchmark():
    if cpu_count() > 1:
        n_docs = 1_000_000

        da_1 = DocumentArray[MyDoc]([MyDoc(title=f'{i}') for i in range(n_docs)])
        start_time = time()
        apply(da=da_1, func=title_da, num_worker=1)
        duration_1_cpu = time() - start_time

        da_2 = DocumentArray[MyDoc]([MyDoc(title=f'{i}') for i in range(n_docs)])
        start_time = time()
        apply(da=da_2, func=title_da, num_worker=2)
        duration_2_cpu = time() - start_time

        assert duration_2_cpu < duration_1_cpu
