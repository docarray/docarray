import pickle

import pytest

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
