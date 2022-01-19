import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.weaviate import DocumentArrayWeaviate


@pytest.fixture
def docarray100():
    yield DocumentArray(Document(text=j) for j in range(100))


@pytest.fixture
def docarrayweaviate100():
    yield DocumentArrayWeaviate(Document(text=j) for j in range(100))


def test_getter_int_str_da(docarray100):
    _test_getter_int_str(docarray100)


def test_getter_int_str_daw(docarrayweaviate100):
    _test_getter_int_str(docarrayweaviate100)


def _test_getter_int_str(docs):
    # getter
    assert docs[99].text == 99
    assert docs[np.int(99)].text == 99
    assert docs[-1].text == 99
    assert docs[0].text == 0
    # string index
    assert docs[docs[0].id].text == 0
    assert docs[docs[99].id].text == 99
    assert docs[docs[-1].id].text == 99

    with pytest.raises(IndexError):
        docs[100]

    with pytest.raises(KeyError):
        docs['adsad']


def test_setter_int_str_da(docarray100):
    _test_getter_int_str(docarray100)


def test_setter_int_str_daw(docarrayweaviate100):
    _test_getter_int_str(docarrayweaviate100)


def _test_setter_int_str(docs):
    # setter
    docs[99] = Document(text='hello')
    docs[0] = Document(text='world')

    assert docs[99].text == 'hello'
    assert docs[-1].text == 'hello'
    assert docs[0].text == 'world'

    docs[docs[2].id] = Document(text='doc2')
    # string index
    assert docs[docs[2].id].text == 'doc2'


def test_del_int_str_da(docarray100):
    _test_del_int_str(docarray100)


def test_del_int_str_daw(docarrayweaviate100):
    _test_del_int_str(docarrayweaviate100)


def _test_del_int_str(docs):
    zero_id = docs[0].id
    del docs[0]
    assert len(docs) == 99
    assert zero_id not in docs

    new_zero_id = docs[0].id
    new_doc_zero = docs[0]
    del docs[new_zero_id]
    assert len(docs) == 98
    assert zero_id not in docs
    assert new_doc_zero not in docs


def test_slice_da(docarray100):
    _test_slice(docarray100)


def test_slice_daw(docarrayweaviate100):
    _test_slice(docarrayweaviate100)


def _test_slice(docs):
    # getter
    assert len(docs[1:5]) == 4
    assert len(docs[1:100:5]) == 20  # 1 to 100, sep with 5

    # setter
    with pytest.raises(TypeError, match='can only assign an iterable'):
        docs[1:5] = Document(text='repl')

    docs[1:5] = [Document(text=f'repl{j}') for j in range(4)]
    for d in docs[1:5]:
        assert d.text.startswith('repl')
    assert len(docs) == 100

    # del
    zero_doc = docs[0]
    twenty_doc = docs[20]
    del docs[0:20]
    assert len(docs) == 80
    assert zero_doc not in docs
    assert twenty_doc in docs


def test_sequence_bool_index_da(docarray100):
    return _test_sequence_bool_index(docarray100)


def test_sequence_bool_index_daw(docarrayweaviate100):
    return _test_sequence_bool_index(docarrayweaviate100)


def _test_sequence_bool_index(docs):
    # getter
    mask = [True, False] * 50
    assert len(docs[mask]) == 50
    assert len(docs[[True, False]]) == 1

    # setter
    mask = [True, False] * 50
    docs[mask] = [Document(text=f'repl{j}') for j in range(50)]

    for idx, d in enumerate(docs):
        if idx % 2 == 0:
            # got replaced
            assert d.text.startswith('repl')
        else:
            assert isinstance(d.text, int)

    assert len(docs) == 100

    # del
    del docs[mask]
    assert len(docs) == 50

    del docs[mask]
    assert len(docs) == 25


@pytest.mark.parametrize('nparray', [lambda x: x, np.array, tuple])
def test_sequence_int_da(docarray100, nparray):
    _test_sequence_int(docarray100, nparray)


@pytest.mark.parametrize('nparray', [lambda x: x, np.array, tuple])
def test_sequence_int_daw(docarrayweaviate100, nparray):
    _test_sequence_int(docarrayweaviate100, nparray)


def _test_sequence_int(docs, nparray):
    # getter
    idx = nparray([1, 3, 5, 7, -1, -2])
    assert len(docs[idx]) == len(idx)

    # setter
    docs[idx] = [Document(text='repl') for _ in range(len(idx))]
    for _id in idx:
        assert docs[_id].text == 'repl'

    # del
    idx = [-3, -4, -5, 9, 10, 11]
    del docs[idx]
    assert len(docs) == 100 - len(idx)


def test_sequence_str_da(docarray100):
    _test_sequence_str(docarray100)


def test_sequence_str_daw(docarrayweaviate100):
    _test_sequence_str(docarrayweaviate100)


def _test_sequence_str(docs):
    # getter
    idx = [d.id for d in docs[1, 3, 5, 7, -1, -2]]

    assert len(docs[idx]) == len(idx)
    assert len(docs[tuple(idx)]) == len(idx)

    # setter
    docs[idx] = [Document(text='repl') for _ in range(len(idx))]
    idx = [d.id for d in docs[1, 3, 5, 7, -1, -2]]
    for _id in idx:
        assert docs[_id].text == 'repl'

    # del
    idx = [d.id for d in docs[-3, -4, -5, 9, 10, 11]]
    del docs[idx]
    assert len(docs) == 100 - len(idx)


def test_docs_list_tuple_da(docarray100):
    _test_docs_list_tuple(docarray100)


def test_docs_list_tuple_daw(docarrayweaviate100):
    _test_docs_list_tuple(docarrayweaviate100)


def _test_docs_list_tuple(docs):
    assert isinstance(docs[99, 98], DocumentArray)
    assert len(docs[99, 98]) == 2


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArrayWeaviate])
def test_path_syntax_indexing(da_cls):
    da = DocumentArray.empty(3)
    for d in da:
        d.chunks = DocumentArray.empty(5)
        d.matches = DocumentArray.empty(7)
        for c in d.chunks:
            c.chunks = DocumentArray.empty(3)
    if da_cls is DocumentArrayWeaviate:
        da = da_cls(da)
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


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArrayWeaviate])
def test_attribute_indexing(da_cls):
    da = da_cls.empty(10)
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


# @pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArrayWeaviate])
# def test_blob_attribute_selector(da_cls):
#     import scipy.sparse
#
#     sp_embed = np.random.random([3, 10])
#     sp_embed[sp_embed > 0.1] = 0
#     sp_embed = scipy.sparse.coo_matrix(sp_embed)
#
#     da = da_cls.empty(3)
#
#     da[:, 'embedding'] = sp_embed
#
#     assert da[:, 'embedding'].shape == (3, 10)
#
#     for d in da:
#         assert d.embedding.shape == (1, 10)
#
#     v1, v2 = da[:, ['embedding', 'id']]
#     assert isinstance(v1, scipy.sparse.coo_matrix)
#     assert isinstance(v2, list)
#
#     v1, v2 = da[:, ['id', 'embedding']]
#     assert isinstance(v2, scipy.sparse.coo_matrix)
#     assert isinstance(v1, list)
#
#
# def test_advance_selector_mixed():
#     da = DocumentArray.empty(10)
#     da.embeddings = np.random.random([10, 3])
#     da.match(da, exclude_self=True)
#
#     assert len(da[:, ('id', 'embedding', 'matches')]) == 3
#     assert len(da[:, ('id', 'embedding', 'matches')][0]) == 10


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArrayWeaviate])
def test_single_boolean_and_padding(da_cls):
    da = da_cls.empty(3)

    with pytest.raises(IndexError):
        da[True]

    with pytest.raises(IndexError):
        da[True] = Document()

    with pytest.raises(IndexError):
        del da[True]

    assert len(da[True, False]) == 1
    assert len(da[False, False]) == 0


@pytest.mark.parametrize('da_cls', [DocumentArray, DocumentArrayWeaviate])
def test_edge_case_two_strings(da_cls):
    # getitem
    da = da_cls([Document(id='1'), Document(id='2'), Document(id='3')])
    assert da['1', 'id'] == '1'
    assert len(da['1', '2']) == 2
    assert isinstance(da['1', '2'], DocumentArray)
    with pytest.raises(KeyError):
        da['hello', '2']
    with pytest.raises(AttributeError):
        da['1', 'hello']
    assert len(da['1', '2', '3']) == 3
    assert isinstance(da['1', '2', '3'], DocumentArray)

    # delitem
    del da['1', '2']
    assert len(da) == 1

    da = da_cls([Document(id='1'), Document(id='2'), Document(id='3')])
    if da_cls != DocumentArrayWeaviate:
        del da['1', 'id']
        assert len(da) == 3
        assert not da[0].id
    else:
        with pytest.raises(
            ValueError, match='cannot pop id from DocumentArrayWeaviate'
        ):
            del da['1', 'id']

    del da['2', 'hello']

    # setitem
    da = da_cls([Document(id='1'), Document(id='2'), Document(id='3')])
    da['1', '2'] = DocumentArray.empty(2)
    assert da[0].id != '1'
    assert da[1].id != '2'

    da = da_cls([Document(id='1'), Document(id='2'), Document(id='3')])
    da['1', 'text'] = 'hello'
    assert da['1'].text == 'hello'

    with pytest.raises(ValueError):
        da['1', 'hellohello'] = 'hello'
