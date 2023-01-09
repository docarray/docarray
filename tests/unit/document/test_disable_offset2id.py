import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.storage.base.helper import Offset2ID


@pytest.fixture(scope='function')
def docs():
    d1 = Document(embedding=np.array([10, 0]))
    d2 = Document(embedding=np.array([0, 10]))
    d3 = Document(embedding=np.array([-10, -10]))
    yield d1, d2, d3


@pytest.mark.parametrize(
    'storage,config',
    [
        ('weaviate', {'n_dim': 2, 'list_like': False}),
        ('annlite', {'n_dim': 2, 'list_like': False}),
        ('qdrant', {'n_dim': 2, 'list_like': False}),
        ('elasticsearch', {'n_dim': 2, 'list_like': False}),
        ('redis', {'n_dim': 2, 'list_like': False}),
    ],
)
def test_disable_offset2id(docs, storage, config, start_storage):
    if config:
        da = DocumentArray(storage=storage, config=config)
    else:
        da = DocumentArray(storage=storage)

    assert len(da) == 0

    da.extend(docs)
    assert len(da) == 3

    with pytest.raises(ValueError):
        da[0]


def test_offset2ids(docs, start_storage):
    list_like = False
    ids = [str(i) for i in range(3)]
    offset2id = Offset2ID(ids, list_like)
    len_before = len(offset2id)
    offset2id.append("4")
    len_after = len(offset2id)
    assert len_before == len_after

    len_before = len(offset2id)
    offset2id.extend("4")
    len_after = len(offset2id)
    assert len_before == len_after

    len_before = len(offset2id)
    offset2id.delete_by_ids("3")
    len_after = len(offset2id)
    assert len_before == len_after

    with pytest.raises(ValueError):
        offset2id.index(2)
    with pytest.raises(ValueError):
        offset2id.get_id("2")
