import pytest

from docarray import DocumentArray


@pytest.fixture
def docs():
    docs = DocumentArray.empty(5)
    docs[0].text = 'hello'
    docs[0].tags['name'] = 'hello'
    docs[1].text = 'world'
    docs[1].tags['name'] = 'hello'
    docs[2].tags['x'] = 0.3
    docs[2].tags['y'] = 0.6
    docs[3].tags['x'] = 0.8

    return docs


def test_empty_filter(docs):
    result = docs.find({})
    assert len(result) == 5


def test_simple_filter(docs):
    result = docs.find({'text': {'$eq': 'hello'}})
    assert len(result) == 1
    assert result[0].text == 'hello'

    result = docs.find({'tags__x': {'$gte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['x'] == 0.8

    result = docs.find({'tags__name': {'$regex': '^h'}})
    assert len(result) == 2
    assert result[1].id == docs[1].id

    result = docs.find({'text': {'$regex': '^h'}})
    assert len(result) == 1
    assert result[0].id == docs[0].id

    result = docs.find({'tags': {'$size': 2}})
    assert result[0].id == docs[2].id

    result = docs.find({'text': {'$exists': True}})
    assert len(result) == 2
    result = docs.find({'tensor': {'$exists': True}})
    assert len(result) == 0


def test_logic_filter(docs):
    result = docs.find({'$or': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 2
    assert result[0].tags['x'] == 0.3 and result[1].tags['x'] == 0.8

    result = docs.find({'$or': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 2
    assert result[0].tags['x'] == 0.3

    result = docs.find({'tags__x': {'$gte': 0.1, '$lte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['y'] == 0.6

    result = docs.find({'$and': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 1
    assert result[0].tags['y'] == 0.6

    result = docs.find({'$not': {'tags__x': {'$gte': 0.5}}})
    assert len(result) == 4
    assert 'x' not in result[0].tags or result[0].tags['x'] < 0.5

    result = docs.find({'$not': {'tags__x': {'$gte': 0.1}, 'tags__y': {'$gte': 0.5}}})
    assert len(result) == 4


def test_placehold_filter(docs):
    result = docs.find({'text': {'$eq': '{tags__name}'}})
    assert len(result) == 1
    assert result[0].id == docs[0].id
