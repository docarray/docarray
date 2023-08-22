import string
import random

import pytest
import time
import os


cur_dir = os.path.dirname(os.path.abspath(__file__))
milvus_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='session', autouse=True)
def start_storage():
    os.system(f"docker compose -f {milvus_yml} up -d --remove-orphans")
    time.sleep(2)

    yield
    os.system(f"docker compose -f {milvus_yml} down --remove-orphans")


@pytest.fixture(scope='function')
def tmp_index_name():
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for _ in range(15))
    return random_string
