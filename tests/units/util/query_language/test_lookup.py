import pytest
from docarray.utils.query_language.lookup import point_get, lookup


class A:
    class B:
        c = 0
        d = 'docarray'
        e = [0, 1]
        f = {}

    b: B = B()


@pytest.mark.parametrize('input', [A(), {'b': {'c': 0, 'd': 'docarray'}}])
def test_point_get(input):
    assert point_get(input, 'b.c') == 0

    expected_exception = KeyError if isinstance(input, dict) else AttributeError
    with pytest.raises(expected_exception):
        _ = point_get(input, 'z')

    with pytest.raises(expected_exception):
        _ = point_get(input, 'b.z')


@pytest.mark.parametrize(
    'input', [A(), {'b': {'c': 0, 'd': 'docarray', 'e': [0, 1], 'f': {}}}]
)
def test_lookup(input):
    assert lookup('b.c__exact', 0, input)
    assert not lookup('b.c__gt', 0, input)
    assert lookup('b.c__gte', 0, input)
    assert not lookup('b.c__lt', 0, input)
    assert lookup('b.c__lte', 0, input)
    assert lookup('b.d__regex', 'array*', input)
    assert lookup('b.d__contains', 'array', input)
    assert lookup('b.d__icontains', 'Array', input)
    assert lookup('b.d__in', ['a', 'docarray'], input)
    assert lookup('b.d__nin', ['a', 'b'], input)
    assert lookup('b.d__startswith', 'doc', input)
    assert lookup('b.d__istartswith', 'Doc', input)
    assert lookup('b.d__endswith', 'array', input)
    assert lookup('b.d__iendswith', 'Array', input)
    assert lookup('b.e__size', 2, input)
    assert not lookup('b.e__size', 3, input)
    assert lookup('b.d__size', len('docarray'), input)
    assert not lookup('b.e__size', len('docarray') + 1, input)
    assert not lookup('b.z__exists', True, input)
    assert lookup('b.z__exists', False, input)
    assert not lookup('b.f.z__exists', True, input)
    assert lookup('b.f.z__exists', False, input)
