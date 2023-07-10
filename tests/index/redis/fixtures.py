import os
import time
import uuid
import pytest


@pytest.fixture(scope='session', autouse=True)
def start_redis():
    os.system(
        'docker run --name redis-stack-server -p 6379:6379 -d redis/redis-stack-server:7.2.0-RC2'
    )
    time.sleep(1)

    yield

    os.system('docker rm -f redis-stack-server')


@pytest.fixture(scope='function')
def tmp_index_name():
    return uuid.uuid4().hex
