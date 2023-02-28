import pytest

from docarray import DocumentArray
from docarray.documents import Text


def get_test_da(n: int):
    return DocumentArray[Text](Text(text=f'text {i}') for i in range(n))


@pytest.mark.skip('Test env has no jina api key?')
def test_pushpull():
    da1 = get_test_da(2**8)
    da1.push('jinaai://test-da', public=True, show_progress=True)
    da2 = DocumentArray[Text].pull('jinaai://test-da', show_progress=True)
    assert len(da1) == len(da2)
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))


@pytest.mark.skip('Not implemented yet')
def test_pushpull_stream():
    da1 = get_test_da(2**8)
    DocumentArray[Text].push_stream(
        iter(da1), 'jinaai://test-da-stream', public=True, show_progress=True
    )
    da2 = DocumentArray[Text].pull_stream('jinaai://test-da-stream', show_progress=True)

    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    with pytest.raises(StopIteration):
        next(da2)
