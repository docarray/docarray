import numpy as np
import pytest

from docarray import DocumentArray, Document


@pytest.fixture
def docarray100():
    yield DocumentArray(Document(text=j) for j in range(100))


def test_getter_int_str(docarray100):
    # getter
    assert docarray100[99].text == 99
    assert docarray100[np.int(99)].text == 99
    assert docarray100[-1].text == 99
    assert docarray100[0].text == 0
    # string index
    assert docarray100[docarray100[0].id].text == 0
    assert docarray100[docarray100[99].id].text == 99
    assert docarray100[docarray100[-1].id].text == 99

    with pytest.raises(IndexError):
        docarray100[100]

    with pytest.raises(KeyError):
        docarray100['adsad']


def test_setter_int_str(docarray100):
    # setter
    docarray100[99] = Document(text='hello')
    docarray100[0] = Document(text='world')

    assert docarray100[99].text == 'hello'
    assert docarray100[-1].text == 'hello'
    assert docarray100[0].text == 'world'

    docarray100[docarray100[2].id] = Document(text='doc2')
    # string index
    assert docarray100[docarray100[2].id].text == 'doc2'


def test_del_int_str(docarray100):
    zero_id = docarray100[0].id
    del docarray100[0]
    assert len(docarray100) == 99
    assert zero_id not in docarray100

    new_zero_id = docarray100[0].id
    new_doc_zero = docarray100[0]
    del docarray100[new_zero_id]
    assert len(docarray100) == 98
    assert zero_id not in docarray100
    assert new_doc_zero not in docarray100


def test_slice(docarray100):
    # getter
    assert len(docarray100[1:5]) == 4
    assert len(docarray100[1:100:5]) == 20  # 1 to 100, sep with 5

    # setter
    with pytest.raises(TypeError, match='can only assign an iterable'):
        docarray100[1:5] = Document(text='repl')

    docarray100[1:5] = [Document(text=f'repl{j}') for j in range(4)]
    for d in docarray100[1:5]:
        assert d.text.startswith('repl')
    assert len(docarray100) == 100

    # del
    zero_doc = docarray100[0]
    twenty_doc = docarray100[20]
    del docarray100[0:20]
    assert len(docarray100) == 80
    assert zero_doc not in docarray100
    assert twenty_doc in docarray100


def test_sequence_bool_index(docarray100):
    # getter
    mask = [True, False] * 50
    assert len(docarray100[mask]) == 50
    assert len(docarray100[[True, False]]) == 1

    # setter
    mask = [True, False] * 50
    docarray100[mask] = [Document(text=f'repl{j}') for j in range(50)]

    for idx, d in enumerate(docarray100):
        if idx % 2 == 0:
            # got replaced
            assert d.text.startswith('repl')
        else:
            assert isinstance(d.text, int)

    # del
    del docarray100[mask]
    assert len(docarray100) == 50

    del docarray100[mask]
    assert len(docarray100) == 25


@pytest.mark.parametrize('nparray', [lambda x: x, np.array, tuple])
def test_sequence_int(docarray100, nparray):
    # getter
    idx = nparray([1, 3, 5, 7, -1, -2])
    assert len(docarray100[idx]) == len(idx)

    # setter
    docarray100[idx] = [Document(text='repl') for _ in range(len(idx))]
    for _id in idx:
        assert docarray100[_id].text == 'repl'

    # del
    idx = [-3, -4, -5, 9, 10, 11]
    del docarray100[idx]
    assert len(docarray100) == 100 - len(idx)


def test_sequence_str(docarray100):
    # getter
    idx = [d.id for d in docarray100[1, 3, 5, 7, -1, -2]]

    assert len(docarray100[idx]) == len(idx)
    assert len(docarray100[tuple(idx)]) == len(idx)

    # setter
    docarray100[idx] = [Document(text='repl') for _ in range(len(idx))]
    idx = [d.id for d in docarray100[1, 3, 5, 7, -1, -2]]
    for _id in idx:
        assert docarray100[_id].text == 'repl'

    # del
    idx = [d.id for d in docarray100[-3, -4, -5, 9, 10, 11]]
    del docarray100[idx]
    assert len(docarray100) == 100 - len(idx)


def test_docarray_list_tuple(docarray100):
    assert isinstance(docarray100[99, 98], DocumentArray)
    assert len(docarray100[99, 98]) == 2
