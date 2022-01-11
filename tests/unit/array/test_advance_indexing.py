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


def test_path_syntax_indexing():
    da = DocumentArray().empty(3)
    for d in da:
        d.chunks = DocumentArray.empty(5)
        d.matches = DocumentArray.empty(7)
        for c in d.chunks:
            c.chunks = DocumentArray.empty(3)
    assert len(da['@c']) == 3 * 5
    assert len(da['@c:1']) == 3
    assert len(da['@c-1:']) == 3
    assert len(da['@c1']) == 3
    assert len(da['@c-2:']) == 3 * 2
    assert len(da['@c1:3']) == 3 * 2
    assert len(da['@c1:3c']) == (3 * 2) * 3
    assert len(da['@c1:3,c1:3c']) == (3 * 2) + (3 * 2) * 3
    assert len(da['@c 1:3 , c 1:3 c']) == (3 * 2) + (3 * 2) * 3
    assert len(da['@cc']) == 3 * 5 * 3
    assert len(da['@cc,m']) == 3 * 5 * 3 + 3 * 7
    assert len(da['@r:1cc,m']) == 1 * 5 * 3 + 3 * 7


def test_attribute_indexing():
    da = DocumentArray.empty(10)
    for v in da[:, 'id']:
        assert v
    da[:, 'mime_type'] = [f'type {j}' for j in range(10)]
    for v in da[:, 'mime_type']:
        assert v
    del da[:, 'mime_type']
    for v in da[:, 'mime_type']:
        assert not v

    da[:, ['text', 'mime_type']] = [
        [f'hello {j}' for j in range(10)],
        [f'type {j}' for j in range(10)],
    ]
    da.summary()

    for v in da[:, ['mime_type', 'text']]:
        for vv in v:
            assert vv


def test_blob_attribute_selector():
    import scipy.sparse

    sp_embed = np.random.random([3, 10])
    sp_embed[sp_embed > 0.1] = 0
    sp_embed = scipy.sparse.coo_matrix(sp_embed)

    da = DocumentArray.empty(3)

    da[:, 'embedding'] = sp_embed

    assert da[:, 'embedding'].shape == (3, 10)

    for d in da:
        assert d.embedding.shape == (1, 10)

    v1, v2 = da[:, ['embedding', 'id']]
    assert isinstance(v1, scipy.sparse.coo_matrix)
    assert isinstance(v2, list)

    v1, v2 = da[:, ['id', 'embedding']]
    assert isinstance(v2, scipy.sparse.coo_matrix)
    assert isinstance(v1, list)


def test_advance_selector_mixed():
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([10, 3])
    da.match(da, exclude_self=True)

    assert len(da[:, ('id', 'embedding', 'matches')]) == 3
    assert len(da[:, ('id', 'embedding', 'matches')][0]) == 10


def test_single_boolean_and_padding():
    from docarray import DocumentArray

    da = DocumentArray.empty(3)

    with pytest.raises(IndexError):
        da[True]

    with pytest.raises(IndexError):
        da[True] = Document()

    with pytest.raises(IndexError):
        del da[True]

    assert len(da[True, False]) == 1
    assert len(da[False, False]) == 0
