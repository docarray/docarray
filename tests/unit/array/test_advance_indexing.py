import os
import time

import numpy as np
import pytest

from docarray import DocumentArray, Document


@pytest.fixture
def docs():
    yield (Document(text=j) for j in range(100))

@pytest.fixture
def indices():
    yield (i for i in [-2,0,2])


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_getter_int_str(docs, storage):
    docs = DocumentArray(docs, storage=storage)
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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_setter_int_str(docs, storage):
    docs = DocumentArray(docs, storage=storage)
    # setter
    docs[99] = Document(text='hello')
    docs[0] = Document(text='world')

    assert docs[99].text == 'hello'
    assert docs[-1].text == 'hello'
    assert docs[0].text == 'world'

    docs[docs[2].id] = Document(text='doc2')
    # string index
    assert docs[docs[2].id].text == 'doc2'


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_del_int_str(docs, storage, indices):

    docs = DocumentArray(docs, storage=storage)
    initial_len = len(docs)
    deleted_elements = 0
    for pos in indices:
        pos_id = docs[pos].id
        del docs[pos]
        deleted_elements += 1
        assert pos_id not in docs
        assert len(docs) == initial_len - deleted_elements

        new_pos_id = docs[pos].id
        new_doc_zero = docs[pos]
        del docs[new_pos_id]
        deleted_elements += 1
        assert len(docs) == initial_len - deleted_elements
        assert pos_id not in docs
        assert new_doc_zero not in docs


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_slice(docs, storage):
    docs = DocumentArray(docs, storage=storage)
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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_sequence_bool_index(docs, storage):
    docs = DocumentArray(docs, storage=storage)
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

    # del
    del docs[mask]
    assert len(docs) == 50

    del docs[mask]
    assert len(docs) == 25


@pytest.mark.parametrize('nparray', [lambda x: x, np.array, tuple])
@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_sequence_int(docs, nparray, storage):
    docs = DocumentArray(docs, storage=storage)
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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_sequence_str(docs, storage):
    docs = DocumentArray(docs, storage=storage)
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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_docarray_list_tuple(docs, storage):
    docs = DocumentArray(docs, storage=storage)
    assert isinstance(docs[99, 98], DocumentArray)
    assert len(docs[99, 98]) == 2


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_path_syntax_indexing(storage):
    da = DocumentArray.empty(3)
    for d in da:
        d.chunks = DocumentArray.empty(5)
        d.matches = DocumentArray.empty(7)
        for c in d.chunks:
            c.chunks = DocumentArray.empty(3)

    if storage == 'sqlite':
        da = DocumentArray(da, storage=storage)
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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_attribute_indexing(storage):
    da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(10))

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


# TODO: enable weaviate storage test
@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_tensor_attribute_selector(storage):
    import scipy.sparse

    sp_embed = np.random.random([3, 10])
    sp_embed[sp_embed > 0.1] = 0
    sp_embed = scipy.sparse.coo_matrix(sp_embed)

    da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(3))

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


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_advance_selector_mixed(storage):
    da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(10))
    da.embeddings = np.random.random([10, 3])
    da.match(da, exclude_self=True)

    assert len(da[:, ('id', 'embedding', 'matches')]) == 3
    assert len(da[:, ('id', 'embedding', 'matches')][0]) == 10


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_single_boolean_and_padding(storage):
    da = DocumentArray(storage=storage)
    da.extend(DocumentArray.empty(3))

    with pytest.raises(IndexError):
        da[True]

    with pytest.raises(IndexError):
        da[True] = Document()

    with pytest.raises(IndexError):
        del da[True]

    assert len(da[True, False]) == 1
    assert len(da[False, False]) == 0


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_edge_case_two_strings(storage):
    # getitem
    da = DocumentArray(
        [Document(id='1'), Document(id='2'), Document(id='3')], storage=storage
    )
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

    da = DocumentArray(
        [Document(id=str(i), text='hey') for i in range(3)], storage=storage
    )
    del da['1', 'text']
    assert len(da) == 3
    assert not da[1].text

    del da['2', 'hello']

    # setitem
    da = DocumentArray(
        [Document(id='1'), Document(id='2'), Document(id='3')], storage=storage
    )
    da['1', '2'] = DocumentArray.empty(2)
    assert da[0].id != '1'
    assert da[1].id != '2'

    da = DocumentArray(
        [Document(id='1'), Document(id='2'), Document(id='3')], storage=storage
    )
    da['1', 'text'] = 'hello'
    assert da['1'].text == 'hello'

    with pytest.raises(ValueError):
        da['1', 'hellohello'] = 'hello'
