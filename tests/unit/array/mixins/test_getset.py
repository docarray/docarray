import numpy as np
import pytest
import scipy.sparse
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite
from tests import random_docs

rand_array = np.random.random([10, 3])


def da_and_dam():
    rand_docs = random_docs(100)
    da = DocumentArray()
    da.extend(rand_docs)
    das = DocumentArraySqlite(rand_docs)
    return (da, das)


def nested_da_and_dam():
    docs = [
        Document(id='r1', chunks=[Document(id='c1'), Document(id='c2')]),
        Document(id='r2', matches=[Document(id='m1'), Document(id='m2')]),
    ]
    da = DocumentArray()
    da.extend(docs)
    das = DocumentArraySqlite(docs)
    return (da, das)


@pytest.mark.parametrize(
    'array',
    [
        rand_array,
        torch.Tensor(rand_array),
        tf.constant(rand_array),
        csr_matrix(rand_array),
    ],
)
def test_set_embeddings_multi_kind(array):
    da = DocumentArray([Document() for _ in range(10)])
    da.embeddings = array


@pytest.mark.parametrize('da', da_and_dam())
def test_da_get_embeddings(da):
    np.testing.assert_almost_equal(da._get_attributes('embedding'), da.embeddings)
    np.testing.assert_almost_equal(da[:, 'embedding'], da.embeddings)


@pytest.mark.parametrize('da', da_and_dam())
def test_embeddings_setter_da(da):
    emb = np.random.random((100, 128))
    da.embeddings = emb
    np.testing.assert_almost_equal(da.embeddings, emb)

    for x, doc in zip(emb, da):
        np.testing.assert_almost_equal(x, doc.embedding)

    da.embeddings = None
    if hasattr(da, 'flush'):
        da.flush()
    assert not da.embeddings


@pytest.mark.parametrize('da', da_and_dam())
def test_embeddings_wrong_len(da):
    embeddings = np.ones((2, 10))

    with pytest.raises(ValueError):
        da.embeddings = embeddings


@pytest.mark.parametrize('da', da_and_dam())
def test_tensors_getter_da(da):
    tensors = np.random.random((100, 10, 10))
    da.tensors = tensors
    assert len(da) == 100
    np.testing.assert_almost_equal(da.tensors, tensors)

    da.tensors = None
    assert da.tensors is None


@pytest.mark.parametrize('da', da_and_dam())
def test_texts_getter_da(da):
    assert len(da.texts) == 100
    assert da.texts == da[:, 'text']
    texts = ['text' for _ in range(100)]
    da.texts = texts
    assert da.texts == texts

    for x, doc in zip(texts, da):
        assert x == doc.text

    da.texts = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert not da.texts


@pytest.mark.parametrize('da', da_and_dam())
def test_setter_by_sequences_in_selected_docs_da(da):
    da[[0, 1, 2], 'text'] = 'test'
    assert da[[0, 1, 2], 'text'] == ['test', 'test', 'test']

    da[[3, 4], 'text'] = ['test', 'test']
    assert da[[3, 4], 'text'] == ['test', 'test']

    da[[5], 'text'] = 'test'
    assert da[[5], 'text'] == ['test']

    da[[6], 'text'] = ['test']
    assert da[[6], 'text'] == ['test']

    # test that ID not present in da works
    da[[0], 'id'] = '999'
    assert ['999'] == da[[0], 'id']

    da[[0, 1], 'id'] = ['101', '102']
    assert ['101', '102'] == da[[0, 1], 'id']


@pytest.mark.parametrize('da', da_and_dam())
def test_texts_wrong_len(da):
    texts = ['hello']

    with pytest.raises(ValueError):
        da.texts = texts


@pytest.mark.parametrize('da', da_and_dam())
def test_tensors_wrong_len(da):
    tensors = np.ones((2, 10, 10))

    with pytest.raises(ValueError):
        da.tensors = tensors


@pytest.mark.parametrize('da', da_and_dam())
def test_blobs_getter_setter(da):
    with pytest.raises(ValueError):
        da.blobs = [b'cc', b'bb', b'aa', b'dd']

    da.blobs = [b'aa'] * len(da)
    assert da.blobs == [b'aa'] * len(da)

    da.blobs = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert not da.blobs


@pytest.mark.parametrize('da', nested_da_and_dam())
def test_ellipsis_getter(da):
    flattened = da[...]
    assert len(flattened) == 6
    for d, doc_id in zip(flattened, ['c1', 'c2', 'r1', 'm1', 'm2', 'r2']):
        assert d.id == doc_id


def test_zero_embeddings():
    a = np.zeros([10, 6])
    da = DocumentArray.empty(10)

    # all zero, dense
    da.embeddings = a
    np.testing.assert_almost_equal(da.embeddings, a)
    for d in da:
        assert d.embedding.shape == (6,)

    # all zero, sparse
    sp_a = scipy.sparse.coo_matrix(a)
    da.embeddings = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)

    # near zero, sparse
    a = np.random.random([10, 6])
    a[a > 0.1] = 0
    sp_a = scipy.sparse.coo_matrix(a)
    da.embeddings = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)
