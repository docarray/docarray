from time import time

import numpy as np
import pytest

from docarray import BaseDocument, DocumentArray
from docarray.typing import NdArray
from docarray.utils.apply import apply


class MyMatrix(BaseDocument):
    matrix: NdArray


def sqrt(doc: MyMatrix) -> MyMatrix:
    # some cpu intensive function
    for i in range(3000):
        sqrt_matrix = np.sqrt(doc.matrix)
    return MyMatrix(matrix=sqrt_matrix)


@pytest.mark.benchmark
def test_apply_benchmark():
    def workload(num_workers: int) -> float:
        n_docs = 5
        rng = np.random.RandomState(0)
        matrices = [rng.random(size=(1000, 1000)) for _ in range(n_docs)]
        da = DocumentArray[MyMatrix]([MyMatrix(matrix=m) for m in matrices])
        start_time = time()
        apply(da=da, func=sqrt, num_worker=num_workers)
        return time() - start_time

    time_1_cpu = workload(num_workers=1)
    time_2_cpu = workload(num_workers=2)

    assert time_2_cpu < time_1_cpu
