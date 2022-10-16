from docarray.score import NamedScore


def test_str_representation():
    assert str(NamedScore(value=0.0)) == str({'value': 0.0})
    assert str(NamedScore(value=None)) == str(dict())
    assert str(NamedScore(value=0.3)) == str({'value': 0.3})
