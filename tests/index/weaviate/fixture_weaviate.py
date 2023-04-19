import os
import time

import pytest
import requests
import weaviate

HOST = "http://localhost:8080"


cur_dir = os.path.dirname(os.path.abspath(__file__))
weaviate_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='session', autouse=True)
def start_storage():
    os.system(f"docker-compose -f {weaviate_yml} up -d --remove-orphans")
    _wait_for_weaviate()

    yield
    os.system(f"docker-compose -f {weaviate_yml} down --remove-orphans")


def _wait_for_weaviate():
    while True:
        try:
            response = requests.get(f"{HOST}/v1/.well-known/ready")
            if response.status_code == 200:
                return
            else:
                time.sleep(0.5)
        except requests.exceptions.ConnectionError:
            time.sleep(1)


@pytest.fixture
def weaviate_client(start_storage):
    client = weaviate.Client(HOST)
    client.schema.delete_all()
    yield client
    client.schema.delete_all()
