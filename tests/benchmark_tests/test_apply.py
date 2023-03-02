from time import time

import numpy as np
import pytest

from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.typing import NdArray
from docarray.utils.apply import apply
from tests.units.typing.test_bytes import IMAGE_PATHS

pytestmark = pytest.mark.benchmark


class MyMatrix(BaseDocument):
    matrix: NdArray


def cpu_intensive(doc: MyMatrix) -> MyMatrix:
    # some cpu intensive function
    for i in range(3000):
        sqrt_matrix = np.sqrt(doc.matrix)
    return MyMatrix(matrix=sqrt_matrix)


def test_apply_multiprocessing_benchmark():
    def time_multiprocessing(num_workers: int) -> float:
        n_docs = 5
        rng = np.random.RandomState(0)
        matrices = [rng.random(size=(1000, 1000)) for _ in range(n_docs)]
        da = DocumentArray[MyMatrix]([MyMatrix(matrix=m) for m in matrices])
        start_time = time()
        apply(da=da, func=cpu_intensive, backend='process', num_worker=num_workers)
        return time() - start_time

    time_1_cpu = time_multiprocessing(num_workers=1)
    print(f"time_1_cpu = {time_1_cpu}")
    time_2_cpu = time_multiprocessing(num_workers=2)
    print(f"time_2_cpu = {time_2_cpu}")

    assert time_2_cpu < time_1_cpu


def io_intensive(img: Image) -> Image:
    # some io intensive function: load and set image url
    t = img.url.load()
    img.tensor = t
    return img


def test_apply_multithreading_benchmark():
    def time_multithreading(num_workers: int) -> float:
        n_docs = 100
        da = DocumentArray[Image](
            [Image(url=IMAGE_PATHS['png']) for _ in range(n_docs)]
        )
        start_time = time()
        apply(da=da, func=io_intensive, backend='thread', num_worker=num_workers)
        return time() - start_time

    time_1_thread = time_multithreading(num_workers=1)
    print(f"time_1_thread = {time_1_thread}")
    time_2_thread = time_multithreading(num_workers=2)
    print(f"time_2_thread = {time_2_thread}")

    assert time_2_thread < time_1_thread
