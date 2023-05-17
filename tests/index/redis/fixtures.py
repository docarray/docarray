import os
import time
import uuid
import pytest
import redis


@pytest.fixture(scope='session', autouse=True)
def start_redis():
    os.system('docker run -d -p 6379:6379 --name test-redis redis')
    time.sleep(1)

    yield

    os.system('docker rm -f test-redis')


@pytest.fixture(scope='function')
def tmp_collection_name():
    return uuid.uuid4().hex


@pytest.fixture
def redis_client():
    """This fixture provides a Redis client"""
    client = redis.Redis(host='localhost', port=6379)
    yield client
    client.flushall()


@pytest.fixture
def redis_config(redis_client):
    """This fixture provides the Redis client and flushes all data after each test case"""
    return redis_client
