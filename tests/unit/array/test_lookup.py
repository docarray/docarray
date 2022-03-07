import pytest
from docarray import Document
from docarray.array.queryset.lookup import lookup


@pytest.fixture
def doc():
    d = Document(
        text='test',
        tags={'x': 0.1, 'y': 1.5, 'name': 'test', 'labels': ['a', 'b', 'test']},
    )
    return d


def test_lookup_ops(doc):
    assert lookup('text__exact', 'test', doc)
    assert lookup('tags__x__neq', 0.2, doc)
    assert lookup('tags__labels__contains', 'a', doc)
    assert not lookup('tags__labels__contains', 'c', doc)
    assert lookup('tags__name__in', ['test'], doc)
    assert lookup('tags__x__nin', [0.2, 0.3], doc)
    assert lookup('tags__name__startswith', 'test', doc)
    assert not lookup('tags__name__startswith', 'Test', doc)
    assert lookup('tags__name__istartswith', 'Test', doc)
    assert lookup('tags__name__endswith', 'test', doc)
    assert not lookup('tags__name__endswith', 'Test', doc)
    assert lookup('tags__name__iendswith', 'Test', doc)
    assert lookup('tags__x__gte', 0.1, doc)
    assert not lookup('tags__y__gt', 1.5, doc)
    assert lookup('tags__x__lte', 0.1, doc)
    assert not lookup('tags__y__lt', 1.5, doc)


def test_lookup_pl(doc):
    assert lookup('tags__x__lt', '{tags__y}', doc)
    assert lookup('text__exact', '{tags__name}', doc)
    assert lookup('text__exact', '{tags__name}', doc)
    assert lookup('text__in', '{tags__labels}', doc)
