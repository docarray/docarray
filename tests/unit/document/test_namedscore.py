import pytest

from docarray.score import NamedScore


@pytest.mark.parametrize(
    'init_args', [None, dict(value=123, description='hello'), NamedScore()]
)
@pytest.mark.parametrize('copy', [True, False])
def test_construct_ns(init_args, copy):
    NamedScore(init_args, copy)
