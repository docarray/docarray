from .fixtures import *  # noqa: F403
from .helpers import assert_when_ready


# move
def test_num_docs(simple_index_with_docs):  # noqa: F811
    index, docs = simple_index_with_docs

    def pred():
        assert index.num_docs() == 10

    assert_when_ready(pred)


# Currently, pymongo cannot create atlas vector search indexes.
def test_configure_index(simple_index):  # noqa: F811
    pass
