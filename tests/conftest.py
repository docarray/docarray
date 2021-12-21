import tempfile

import pytest


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    tmpfile = f'jina_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / tmpfile
