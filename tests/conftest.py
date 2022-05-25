import tempfile
import os
import time
from typing import Dict

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(
    os.path.join(cur_dir, 'unit', 'array', 'docker-compose.yml')
)


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    tmpfile = f'docarray_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / tmpfile


@pytest.fixture(scope='session')
def start_storage():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d "
        f"--remove-orphans"
    )
    from elasticsearch import Elasticsearch

    es = Elasticsearch(hosts='http://localhost:9200/')
    while not es.ping():
        time.sleep(0.5)

    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down "
        f"--remove-orphans"
    )


@pytest.fixture(scope='session')
def set_env_vars(request):
    _old_environ = dict(os.environ)
    os.environ.update(request.param)
    yield
    os.environ.clear()
    os.environ.update(_old_environ)
