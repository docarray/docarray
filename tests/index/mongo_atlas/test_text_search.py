from .fixtures import *  # noqa
from .helpers import assert_when_ready


def test_text_search(simple_index_with_docs):
    simple_index, docs = simple_index_with_docs

    query_string = "Python is a valuable skill"
    expected_text = docs[0].text

    def pred():
        docs, _ = simple_index.text_search(
            query=query_string, search_field='text', limit=1
        )
        assert docs[0].text == expected_text

    assert_when_ready(pred)


def test_text_search_batched(simple_index_with_docs, simple_schema):  # noqa: F811

    index, docs = simple_index_with_docs

    queries = ['processing with Python', 'tips', 'for']
    docs, scores = index.text_search_batched(queries, search_field='text', limit=5)

    assert len(docs) == 3
    assert len(docs[0]) == 1
    assert len(docs[1]) == 1
    assert len(docs[2]) == 2
    assert len(scores) == 3
    assert len(scores[0]) == 1
    assert len(scores[1]) == 1
    assert len(scores[2]) == 2
