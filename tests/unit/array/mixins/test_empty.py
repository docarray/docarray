from docarray import DocumentArray


def test_empty_non_zero():
    da = DocumentArray.empty(10)
    assert len(da) == 10
    da = DocumentArray.empty()
    assert len(da) == 0
