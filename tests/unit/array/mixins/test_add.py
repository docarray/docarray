from docarray import DocumentArray


def test_add_doc():
    da1 = DocumentArray.empty(3)
    da2 = DocumentArray.empty(4)
    da3 = DocumentArray.empty(5)

    all_sum = da1 + da2 + da3
    assert len(all_sum) == 12
    assert len(da1) == 3
    assert len(da2) == 4
    assert len(da3) == 5

    da1 += da2
    assert len(all_sum) == 12
    assert len(da1) == 7
    assert len(da2) == 4
    assert len(da3) == 5
