import os
import time

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='module')
def start_weaviate():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d "
        f"--remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down "
        f"--remove-orphans"
    )
