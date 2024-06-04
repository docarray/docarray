import time
from typing import Callable

from pydantic import Field

from docarray import BaseDoc
from docarray.typing import NdArray

N_DIM = 10


class SimpleSchema(BaseDoc):
    text: str = Field(index_name='text_index')
    number: int
    embedding: NdArray[10] = Field(dim=10, index_name="vector_index")


class SimpleDoc(BaseDoc):
    embedding: NdArray[N_DIM] = Field(dim=N_DIM, index_name="vector_index_1")


class NestedDoc(BaseDoc):
    d: SimpleDoc
    embedding: NdArray[N_DIM] = Field(dim=N_DIM, index_name="vector_index")


class FlatSchema(BaseDoc):
    embedding1: NdArray = Field(dim=N_DIM, index_name="vector_index_1")
    # the dim and N_DIM are setted different on propouse. to check the correct handling of n_dim
    embedding2: NdArray[50] = Field(dim=N_DIM, index_name="vector_index_2")


def assert_when_ready(callable: Callable, tries: int = 5, interval: float = 2):
    """
    Retry callable to account for time taken to change data on the cluster
    """
    while True:
        try:
            callable()
        except AssertionError:
            tries -= 1
            if tries == 0:
                raise
            time.sleep(interval)
        else:
            return
