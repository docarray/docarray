import numpy as np
import pytest

from docarray import Document


@pytest.mark.parametrize('init_args', [None, dict(id=123)])
@pytest.mark.parametrize('copy', [True, False])
def test_construct_doc(init_args, copy):
    d = Document(init_args, copy)


def test_doc_hash_identical():
    d1 = Document(text='hello')
    d2 = Document(text='hello')
    assert hash(d1) != hash(d2)
    d1.id = d2.id
    assert hash(d1) == hash(d2)

def test_doc_hash_complicat_content():
    d1 = Document(text='hello', embedding=np.array([1,2,3]))
    hash(d1)
