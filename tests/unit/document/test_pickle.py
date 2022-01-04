import pickle

import pytest

from docarray import Document
from docarray.document.data import DocumentData
from docarray.base import BaseDCType
from tests import random_docs


@pytest.mark.parametrize('cls', BaseDCType.__subclasses__())
def test_pickle_dump_load(cls):
    r = pickle.loads(pickle.dumps(cls()))
    isinstance(r, cls)


def test_pickle_dump_load_real_doc():
    for d in random_docs(10):
        dr = pickle.loads(pickle.dumps(d))
        assert dr == d
        assert dr.embedding is not None
        assert len(dr.chunks) == len(d.chunks)


def test_pickle_rely_on_data_class_and_document_class():
    # TODO (Han): This is not really a designed behavior, but atm I see no harm
    #  of having it, and no real usecases that against it.

    d = Document()
    d.id = 'hello'
    setattr(d, 'foo', 'bar')
    assert getattr(d, 'foo') == 'bar'
    r_d = Document.from_bytes(d.to_bytes(protocol='pickle'))
    assert r_d.id == d.id
    assert getattr(r_d, 'foo') == 'bar'
