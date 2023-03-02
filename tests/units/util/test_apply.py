import pytest

from docarray import DocumentArray
from docarray.documents import Image
from docarray.utils.apply import apply
from tests.units.typing.test_bytes import IMAGE_PATHS


def foo(d: Image) -> Image:
    if d.url is not None:
        d.tensor = d.url.load()
    return d


@pytest.fixture()
def da():
    da = DocumentArray[Image](
        [Image(url=url) for url in IMAGE_PATHS.values() for _ in range(2)]
    )
    return da


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_apply(da, backend):
    for tensor in da.tensor:
        assert tensor is None

    da_applied = apply(da=da, func=foo, backend=backend)

    assert len(da) == len(da_applied)
    for tensor in da_applied.tensor:
        assert tensor is not None


def test_apply_multiprocessing_lambda_func_raise_exception(da):
    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        apply(da=da, func=lambda x: x, backend='process')


def test_apply_multiprocessing_local_func_raise_exception(da):
    def local_func(x):
        return x

    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        apply(da=da, func=local_func, backend='process')


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_check_order(backend):
    da = DocumentArray[Image]([Image(id=i) for i in range(2)])

    da_applied = apply(da=da, func=foo, backend=backend)

    assert len(da) == len(da_applied)
    for id_1, id_2 in zip(da, da_applied):
        assert id_1 == id_2
