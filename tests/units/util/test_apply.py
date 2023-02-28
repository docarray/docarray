import time
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
import pytest

from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.typing import NdArray
from docarray.utils.apply import apply
from tests.units.typing.test_bytes import IMAGE_PATHS


def foo(d: Image) -> Image:
    if d.url is not None:
        d.tensor = d.url.load()
    return d


@pytest.fixture()
def da():
    da = DocumentArray[Image](
        [Image(url=url) for url in IMAGE_PATHS.values() for _ in range(10)]
    )
    return da


def test_apply(da):
    for tensor in da.tensor:
        assert tensor is None

    da_applied = apply(da=da, func=foo)

    assert len(da) == len(da_applied)
    for tensor in da_applied.tensor:
        assert tensor is not None


def test_apply_with_lambda(da):
    for tensor in da.tensor:
        assert tensor is None

    da_applied = apply(da=da, func=lambda x: x)

    assert len(da) == len(da_applied)
    for tensor in da_applied.tensor:
        assert tensor is None


def test_apply_with_local_function(da):
    def local_func(d: Image) -> Image:
        if d.url is not None:
            d.tensor = d.url.load()
        return d

    for tensor in da.tensor:
        assert tensor is None

    da_applied = apply(da=da, func=local_func)

    assert len(da) == len(da_applied)
    for tensor in da_applied.tensor:
        assert tensor is None


class MyDoc(BaseDocument):
    tensor_a: Optional[NdArray]
    tensor_b: Optional[NdArray]
    tensor_matmul: Optional[NdArray]


@pytest.fixture()
def func():
    def matmul(doc):
        if doc.tensor_a is not None and doc.tensor_b is not None:
            doc.tensor_matmul = np.matmul(doc.tensor_a, doc.tensor_b)
        return doc

    return matmul


def matmul(doc):
    if doc.tensor_a is not None and doc.tensor_b is not None:
        doc.tensor_matmul = np.matmul(doc.tensor_a, doc.tensor_b)
    return doc


def test_benchmark(func):
    time_mproc = []
    time_no_mproc = []

    for n_docs in [1, 2]:
        da = DocumentArray[MyDoc](
            [
                MyDoc(
                    tensor_a=np.random.randn(100, 200),
                    tensor_b=np.random.randn(200, 100),
                )
                for _ in range(n_docs)
            ]
        )

        # with multiprocessing
        start_time = time.time()
        apply(da=da, func=func)
        duration_mproc = time.time() - start_time
        time_mproc.append(duration_mproc)

        # without multiprocessing
        start_time = time.time()
        da_no_mproc = DocumentArray[MyDoc]()
        for i, doc in enumerate(da):
            da_no_mproc.append(func(doc))
        duration_no_mproc = time.time() - start_time
        time_no_mproc.append(duration_no_mproc)

    # if more than 1 CPU available, check that when using multiprocessing
    # grows slower with more documents, then without multiprocessing.
    print(f"cpu_count() = {cpu_count()}")
    if cpu_count() > 1:
        growth_factor_mproc = time_mproc[1] / time_mproc[0]
        growth_factor_no_mproc = time_no_mproc[1] / time_no_mproc[0]
        assert growth_factor_mproc < growth_factor_no_mproc
