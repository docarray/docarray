import os
import time
import uuid

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml_v7 = os.path.abspath(os.path.join(cur_dir, 'v7/docker-compose.yml'))
compose_yml_v8 = os.path.abspath(os.path.join(cur_dir, 'v8/docker-compose.yml'))


@pytest.fixture(scope='module', autouse=True)
def start_storage_v7():
    os.system(f"docker-compose -f {compose_yml_v7} up -d --remove-orphans")
    _wait_for_es()

    yield
    os.system(f"docker-compose -f {compose_yml_v7} down --remove-orphans")


@pytest.fixture(scope='module', autouse=True)
def start_storage_v8():
    os.system(f"docker-compose -f {compose_yml_v8} up -d --remove-orphans")
    _wait_for_es()

    yield
    os.system(f"docker-compose -f {compose_yml_v8} down --remove-orphans")


def _wait_for_es():
    from elasticsearch import Elasticsearch

    es = Elasticsearch(hosts='http://localhost:9200/')
    while not es.ping():
        time.sleep(0.5)


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dims=1000)


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dims=10)
    tens_two: NdArray = Field(dims=50)


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class MyImageDoc(ImageDoc):
    embedding: NdArray = Field(dims=128)


@pytest.fixture(scope='function')
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


@pytest.fixture(scope='function')
def ten_flat_docs():
    return [
        FlatDoc(tens_one=np.random.randn(10), tens_two=np.random.randn(50))
        for _ in range(10)
    ]


@pytest.fixture(scope='function')
def ten_nested_docs():
    return [NestedDoc(d=SimpleDoc(tens=np.random.randn(10))) for _ in range(10)]


@pytest.fixture(scope='function')
def ten_deep_nested_docs():
    return [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.randn(10))))
        for _ in range(10)
    ]


@pytest.fixture(scope='function')
def tmp_index_name():
    return uuid.uuid4().hex
