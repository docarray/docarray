import os
from time import time

import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import NdArray
from docarray.utils.map import map_docs, map_docs_batched
from tests.units.typing.test_bytes import IMAGE_PATHS

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


class MyMatrix(BaseDoc):
    matrix: NdArray


def cpu_intensive(doc: MyMatrix) -> MyMatrix:
    # some cpu intensive function
    for i in range(3000):
        sqrt_matrix = np.sqrt(doc.matrix)
    doc.matrix = sqrt_matrix
    return doc


def test_map_docs_multiprocessing():
    if os.cpu_count() > 1:

        def time_multiprocessing(num_workers: int) -> float:
            n_docs = 5
            rng = np.random.RandomState(0)
            matrices = [rng.random(size=(1000, 1000)) for _ in range(n_docs)]
            da = DocList[MyMatrix]([MyMatrix(matrix=m) for m in matrices])
            start_time = time()
            list(
                map_docs(
                    docs=da,
                    func=cpu_intensive,
                    backend='process',
                    num_worker=num_workers,
                )
            )
            return time() - start_time

        time_1_cpu = time_multiprocessing(num_workers=1)
        time_2_cpu = time_multiprocessing(num_workers=2)

        assert time_2_cpu < time_1_cpu


def cpu_intensive_batch(da: DocList[MyMatrix]) -> DocList[MyMatrix]:
    # some cpu intensive function
    for doc in da:
        for i in range(3000):
            sqrt_matrix = np.sqrt(doc.matrix)
        doc.matrix = sqrt_matrix
    return da


def test_map_docs_batched_multiprocessing():
    if os.cpu_count() > 1:

        def time_multiprocessing(num_workers: int) -> float:
            n_docs = 16
            rng = np.random.RandomState(0)
            matrices = [rng.random(size=(1000, 1000)) for _ in range(n_docs)]
            da = DocList[MyMatrix]([MyMatrix(matrix=m) for m in matrices])
            start_time = time()
            list(
                map_docs_batched(
                    docs=da,
                    func=cpu_intensive_batch,
                    batch_size=8,
                    backend='process',
                    num_worker=num_workers,
                )
            )
            return time() - start_time

        time_1_cpu = time_multiprocessing(num_workers=1)
        time_2_cpu = time_multiprocessing(num_workers=2)

        assert time_2_cpu < time_1_cpu


def io_intensive(img: ImageDoc) -> ImageDoc:
    # some io intensive function: load and set image url
    img.tensor = img.url.load()
    return img


def test_map_docs_multithreading():
    def time_multithreading(num_workers: int) -> float:
        n_docs = 100
        da = DocList[ImageDoc](
            [ImageDoc(url=IMAGE_PATHS['png']) for _ in range(n_docs)]
        )
        start_time = time()
        list(
            map_docs(
                docs=da, func=io_intensive, backend='thread', num_worker=num_workers
            )
        )
        return time() - start_time

    time_1_thread = time_multithreading(num_workers=1)
    time_2_thread = time_multithreading(num_workers=2)

    assert time_2_thread < time_1_thread


def io_intensive_batch(da: DocList[ImageDoc]) -> DocList[ImageDoc]:
    # some io intensive function: load and set image url
    for doc in da:
        doc.tensor = doc.url.load()
    return da


def test_map_docs_batched_multithreading():
    def time_multithreading_batch(num_workers: int) -> float:
        n_docs = 100
        da = DocList[ImageDoc](
            [ImageDoc(url=IMAGE_PATHS['png']) for _ in range(n_docs)]
        )
        start_time = time()
        list(
            map_docs_batched(
                docs=da,
                func=io_intensive_batch,
                backend='thread',
                num_worker=num_workers,
                batch_size=10,
            )
        )
        return time() - start_time

    time_1_thread = time_multithreading_batch(num_workers=1)
    time_2_thread = time_multithreading_batch(num_workers=2)

    assert time_2_thread < time_1_thread
