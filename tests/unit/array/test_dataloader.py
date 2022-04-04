import time

import numpy as np

from docarray import DocumentArray


def cpu_job(da):
    time.sleep(2)
    assert da.tensors.shape
    return da


def gpu_job(da):
    time.sleep(1)
    assert da.tensors.shape


def test_dataloader():
    N = 100
    da = DocumentArray.empty(N)
    da.tensors = np.array(255 * np.random.random([N, 32, 32, 3]), dtype=np.uint8)
    da.save_binary('da.protobuf.gz')
    for da in DocumentArray.dataloader(
        'da.protobuf.gz', func=cpu_job, batch_size=64, num_worker=4
    ):
        gpu_job(da)
