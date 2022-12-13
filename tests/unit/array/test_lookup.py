import numpy as np
import pytest

from docarray import Document


@pytest.fixture
def doc():
    d = Document(
        text='test',
        embedding=np.random.random(10),
        tags={
            'v': np.zeros(3),
            'w': 0,
            'x': 0.1,
            'y': 1.5,
            'z': 1,
            'name': 'test',
            'bar': '',
            'labels': ['a', 'b', 'test'],
        },
    )
    return d


def test_lookup_ops(doc):
    from docarray.array.queryset.lookup import lookup

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

    assert lookup('text__regex', '^test', doc)
    assert not lookup('text__regex', '^est', doc)

    assert lookup('tags__size', 8, doc)
    assert lookup('tags__labels__size', 3, doc)

    assert lookup('tags__exists', True, doc)
    assert lookup('tags__z__exists', True, doc)
    assert lookup('tags__v__exists', True, doc)
    assert lookup('tags__w__exists', True, doc)
    assert lookup('tags__foo__exists', False, doc)
    assert lookup('tags__bar__exists', True, doc)
    assert lookup('embedding__exists', True, doc)
    assert lookup('tensor__exists', False, doc)
    assert lookup('blob__exists', False, doc)
    assert lookup('text__exists', True, doc)


def test_lookup_pl(doc):
    from docarray.array.queryset.lookup import lookup

    assert lookup('tags__x__lt', '{tags__y}', doc)
    assert lookup('text__exact', '{tags__name}', doc)
    assert lookup('text__exact', '{tags__name}', doc)
    assert lookup('text__in', '{tags__labels}', doc)


def test_lookup_funcs():
    from docarray.array.queryset import lookup

    assert lookup.dunder_partition('a') == ('a', None)
    assert lookup.dunder_partition('a__b__c') == ('a__b', 'c')

    assert lookup.iff_not_none('a', lambda y: y == 'a')
    assert not lookup.iff_not_none(None, lambda y: y == 'a')

    lookup.guard_str('a') == 'a'
    lookup.guard_list(['a']) == ['a']

    with pytest.raises(lookup.LookupyError):
        lookup.guard_str(0.1)
        lookup.guard_list(0.1)
        lookup.guard_Q(0.1)
