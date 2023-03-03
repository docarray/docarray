from typing import Generator, Optional

import pytest

from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.typing import ImageUrl, NdArray
from docarray.utils.map import map, map_batch
from tests.units.typing.test_bytes import IMAGE_PATHS

N_DOCS = 2


def load_from_doc(d: Image) -> Image:
    if d.url is not None:
        d.tensor = d.url.load()
    return d


@pytest.fixture()
def da():
    da = DocumentArray[Image]([Image(url=IMAGE_PATHS['png']) for _ in range(N_DOCS)])
    return da


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_map(da, backend):
    for tensor in da.tensor:
        assert tensor is None

    docs = list(map(da=da, func=load_from_doc, backend=backend))

    assert len(docs) == N_DOCS
    for doc in docs:
        assert doc.tensor is not None


def test_map_multiprocessing_lambda_func_raise_exception(da):
    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        list(map(da=da, func=lambda x: x, backend='process'))


def test_map_multiprocessing_local_func_raise_exception(da):
    def local_func(x):
        return x

    with pytest.raises(ValueError, match='Multiprocessing does not allow'):
        list(map(da=da, func=local_func, backend='process'))


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_check_order(backend):
    da = DocumentArray[Image]([Image(id=i) for i in range(N_DOCS)])

    docs = list(map(da=da, func=load_from_doc, backend=backend))

    assert len(docs) == N_DOCS
    for i, doc in enumerate(docs):
        assert doc.id == str(i)


def load_from_da(da: DocumentArray) -> DocumentArray:
    for doc in da:
        doc.tensor = doc.url.load()
    return da


class MyImage(BaseDocument):
    tensor: Optional[NdArray]
    url: ImageUrl


@pytest.mark.slow
@pytest.mark.parametrize('n_docs,batch_size', [(10, 5), (10, 8)])
@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_map_batch(n_docs, batch_size, backend):

    da = DocumentArray[MyImage](
        [MyImage(url=IMAGE_PATHS['png']) for _ in range(n_docs)]
    )
    it = map_batch(da=da, func=load_from_da, batch_size=batch_size, backend=backend)
    assert isinstance(it, Generator)

    for batch in it:
        assert isinstance(batch, DocumentArray[MyImage])
