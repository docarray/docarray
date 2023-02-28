from multiprocessing import Pool
from time import time

import pytest

from docarray import BaseDocument, DocumentArray

#
#
# class MyDoc(BaseDocument):
#     tensor_a: Optional[NdArray]
#     tensor_b: Optional[NdArray]
#     tensor_matmul: Optional[NdArray]
#
#
# def matmul(doc):
#     if doc.tensor_a is not None and doc.tensor_b is not None:
#         doc.tensor_matmul = np.matmul(doc.tensor_a, doc.tensor_b)
#     return doc
#
#
# def test_benchmark():
#     # check that time(apply(num_workers=2)) < time(apply(num_workers=1)) < without multiprocessing
#     time_mp_1 = []
#     time_mp_2 = []
#     time_mp_4 = []
#     time_mp_8 = []
#     time_no_mp = []
#
#     for n_docs in [100, 1_000_000]:
#         da = DocumentArray[MyDoc](
#             [
#                 MyDoc(
#                     tensor_a=np.random.randn(10, 20),
#                     tensor_b=np.random.randn(20, 10),
#                 )
#                 for _ in range(n_docs)
#             ]
#         )
#
#         # with multiprocessing
#         start_time = time()
#         apply(da=da, func=matmul, num_worker=1)
#         duration_mp = time() - start_time
#         time_mp_1.append(duration_mp)
#
#         da = DocumentArray[MyDoc](
#             [
#                 MyDoc(
#                     tensor_a=np.random.randn(10, 20),
#                     tensor_b=np.random.randn(20, 10),
#                 )
#                 for _ in range(n_docs)
#             ]
#         )
#         # with multiprocessing
#         start_time = time()
#         apply(da=da, func=matmul, num_worker=2)
#         duration_mp = time() - start_time
#         time_mp_2.append(duration_mp)
#
#         # da = DocumentArray[MyDoc](
#         #     [
#         #         MyDoc(
#         #             tensor_a=np.random.randn(10, 20),
#         #             tensor_b=np.random.randn(20, 10),
#         #         )
#         #         for _ in range(n_docs)
#         #     ]
#         # )
#         #
#         # # with multiprocessing
#         # start_time = time()
#         # apply(da=da, func=matmul, num_worker=4)
#         # duration_mp = time() - start_time
#         # time_mp_4.append(duration_mp)
#         #
#         # da = DocumentArray[MyDoc](
#         #     [
#         #         MyDoc(
#         #             tensor_a=np.random.randn(10, 20),
#         #             tensor_b=np.random.randn(20, 10),
#         #         )
#         #         for _ in range(n_docs)
#         #     ]
#         # )
#         # # with multiprocessing
#         # start_time = time()
#         # apply(da=da, func=matmul, num_worker=8)
#         # duration_mp = time() - start_time
#         # time_mp_8.append(duration_mp)
#         #
#         # da = DocumentArray[MyDoc](
#         #     [
#         #         MyDoc(
#         #             tensor_a=np.random.randn(10, 20),
#         #             tensor_b=np.random.randn(20, 10),
#         #         )
#         #         for _ in range(n_docs)
#         #     ]
#         # )
#         # # without multiprocessing
#         # start_time = time()
#         # da_no_mp = DocumentArray[MyDoc]()
#         # for i, doc in enumerate(da):
#         #     da_no_mp.append(matmul(doc))
#         # duration_no_mp = time() - start_time
#         # time_no_mp.append(duration_no_mp)
#
#     # if more than 1 CPU available, check that when using multiprocessing
#     # grows slower with more documents, then without multiprocessing.
#     print(f"cpu_count() = {cpu_count()}")
#     if cpu_count() > 1:
#         slope_mp_1 = time_mp_1[1] / time_mp_1[0]
#         print(f"\ntime_mp_1 = {time_mp_1}")
#         print(f"slope_mp_1 = {slope_mp_1}")
#
#         slope_mp_2 = time_mp_2[1] / time_mp_2[0]
#         print(f"\ntime_mp_2 = {time_mp_2}")
#         print(f"slope_mp_2 = {slope_mp_2}")
#
#         # slope_mp_4 = time_mp_4[1] / time_mp_4[0]
#         # print(f"\ntime_mp_4 = {time_mp_4}")
#         # print(f"slope_mp_4 = {slope_mp_4}")
#         #
#         # slope_mp_8 = time_mp_8[1] / time_mp_8[0]
#         # print(f"\ntime_mp_8 = {time_mp_8}")
#         # print(f"slope_mp_8 = {slope_mp_8}")
#         #
#         # slope_no_mp = time_no_mp[1] / time_no_mp[0]
#         # print(f"\ntime_no_mp = {time_no_mp}")
#         # print(f"slope_no_mp = {slope_no_mp}")
#         # # assert slope_mp * 10 < slope_no_mp


class MyDoc(BaseDocument):
    number: int


def square(x: int) -> int:
    return x * x


def square_da(doc: MyDoc) -> MyDoc:
    doc.number = doc.number * doc.number
    return doc


@pytest.mark.benchmark
def test_apply_benchmark():
    n_docs = 1_000_000
    print(f"n_docs = {n_docs}")

    # without docarray
    numbers = [i for i in range(n_docs)]
    start_time = time()
    with Pool(1) as p_1:
        p_1.imap(square, numbers)
    duration = time() - start_time
    print(f"\nduration 1 cpu    = {duration}")

    numbers = [i for i in range(n_docs)]
    start_time = time()
    with Pool(2) as p_2:
        p_2.imap(square, numbers)
    duration = time() - start_time
    print(f"duration 2 cpu    = {duration}")

    # with docarray
    da_1 = DocumentArray[MyDoc]([MyDoc(number=i) for i in range(n_docs)])
    start_time = time()
    with Pool(1) as p_3:
        p_3.imap(square_da, da_1)
    duration = time() - start_time
    print(f"\nduration 1 cpu da = {duration}")

    da_2 = DocumentArray[MyDoc]([MyDoc(number=i) for i in range(n_docs)])
    start_time = time()
    with Pool(2) as p_4:
        p_4.imap(square_da, da_2)
    duration = time() - start_time
    print(f"duration 2 cpu da = {duration}")
