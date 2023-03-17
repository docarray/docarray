import os
import time

import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.doc_index]


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dims=1000)


class FlatDoc(BaseDocument):
    tens_one: NdArray = Field(dims=10)
    tens_two: NdArray = Field(dims=50)


class NestedDoc(BaseDocument):
    d: SimpleDoc


class DeepNestedDoc(BaseDocument):
    d: NestedDoc


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
