import pytest

from docarray import DocumentArray


@pytest.fixture
def docs():
    docs = DocumentArray.empty(5)
    docs[0].text = 'hello'
    docs[1].text = 'world'
    docs[2].tags['x'] = 0.3
    docs[2].tags['y'] = 0.6
    docs[3].tags['x'] = 0.8
    return docs


def test_empty_filter(docs):
    result = docs.find({})
    assert len(result) == 5


def test_sample_filter(docs):
    result = docs.find({'text': {'$eq': 'hello'}})
    assert len(result) == 1
    assert result[0].text == 'hello'

    result = docs.find({'tags__x': {'$gte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['x'] == 0.8


def test_logic_filter(docs):
    result = docs.find({'$or': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 2
    assert result[0].tags['x'] == 0.3 and result[1].tags['x'] == 0.8

    result = docs.find(
        {'$or': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}}, limit=1
    )
    assert len(result) == 1
    assert result[0].tags['x'] == 0.3

    result = docs.find({'tags__x': {'$gte': 0.1, '$lte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['y'] == 0.6

    result = docs.find({'$and': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 1
    assert result[0].tags['y'] == 0.6
