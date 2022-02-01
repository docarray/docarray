import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite


def da_and_dam():
    da = DocumentArray(
        [
            Document(text='hello'),
            Document(text='hello world'),
            Document(text='goodbye world!'),
        ]
    )

    das = DocumentArraySqlite(
        [
            Document(text='hello'),
            Document(text='hello world'),
            Document(text='goodbye world!'),
        ]
    )

    return (da, das)


@pytest.mark.parametrize('min_freq', [1, 2, 3])
@pytest.mark.parametrize('da', da_and_dam())
def test_da_vocabulary(da, min_freq):
    vocab = da.get_vocabulary(min_freq)
    if min_freq <= 1:
        assert set(vocab.values()) == {2, 3, 4}  # 0,1 are reserved
        assert set(vocab.keys()) == {'hello', 'world', 'goodbye'}
    elif min_freq == 2:
        assert set(vocab.values()) == {2, 3}  # 0,1 are reserved
        assert set(vocab.keys()) == {'hello', 'world'}
    elif min_freq == 3:
        assert not vocab.values()
        assert not vocab.keys()


@pytest.mark.parametrize('test_docs', da_and_dam())
def test_da_text_to_tensor_non_max_len(test_docs):
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab))
    np.testing.assert_array_equal(test_docs[0].tensor, [2])
    np.testing.assert_array_equal(test_docs[1].tensor, [2, 3])
    np.testing.assert_array_equal(test_docs[2].tensor, [4, 3])
    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'hello world'
    assert test_docs[2].text == 'goodbye world'


@pytest.mark.parametrize('test_docs', da_and_dam())
def test_da_text_to_tensor_max_len_3(test_docs):
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=3))

    np.testing.assert_array_equal(test_docs[0].tensor, [0, 0, 2])
    np.testing.assert_array_equal(test_docs[1].tensor, [0, 2, 3])
    np.testing.assert_array_equal(test_docs[2].tensor, [0, 4, 3])

    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'hello world'
    assert test_docs[2].text == 'goodbye world'


@pytest.mark.parametrize('test_docs', da_and_dam())
def test_da_text_to_tensor_max_len_1(test_docs):
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=1))

    np.testing.assert_array_equal(test_docs[0].tensor, [2])
    np.testing.assert_array_equal(test_docs[1].tensor, [3])
    np.testing.assert_array_equal(test_docs[2].tensor, [3])

    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'world'
    assert test_docs[2].text == 'world'


@pytest.mark.parametrize('da', da_and_dam())
def test_convert_text_tensor_random_text(da):
    texts = ['a short phrase', 'word', 'this is a much longer sentence']
    da.clear()
    da.extend(Document(text=t) for t in texts)
    vocab = da.get_vocabulary()

    # encoding
    da.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=10))

    # decoding
    da.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert texts
    assert da.texts == texts
