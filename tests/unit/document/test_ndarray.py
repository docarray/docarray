import numpy as np
import paddle
import pytest
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, csc_matrix

from docarray import Document


def get_ndarrays():
    a = np.random.random([10, 3])
    a[a > 0.5] = 0
    return [
        (a, False),
        (torch.tensor(a), False),
        (tf.constant(a), False),
        (paddle.to_tensor(a), False),
        (torch.tensor(a).to_sparse(), True),
        (tf.sparse.from_dense(a), True),
        (csr_matrix(a), True),
        (bsr_matrix(a), True),
        (coo_matrix(a), True),
        (csc_matrix(a), True),
    ]


@pytest.mark.parametrize('ndarray_val, is_sparse', get_ndarrays())
@pytest.mark.parametrize('attr', ['embedding', 'tensor'])
def test_ndarray_force_numpy(ndarray_val, attr, is_sparse):
    d = Document()
    setattr(d, attr, ndarray_val)
    assert type(getattr(Document.from_protobuf(d.to_protobuf()), attr)) is type(
        ndarray_val
    )


def get_wrong_embeddings():
    a = np.zeros((5, 100))
    return [
        a,
        # torch.tensor(a),
        # tf.constant(a),
        # paddle.to_tensor(a),
        # torch.tensor(a).to_sparse(),
        # csr_matrix(a),
        # bsr_matrix(a),
        # coo_matrix(a),
        # csc_matrix(a),
    ]


@pytest.mark.parametrize('ndarray_val', get_wrong_embeddings())
def test_wrong_embedding_shape_init(ndarray_val):

    with pytest.warns(UserWarning, match='embedding should be a vector') as record:
        Document(embedding=ndarray_val)

    assert len(record) == 1


@pytest.mark.parametrize('ndarray_val', get_wrong_embeddings())
def test_wrong_embedding_shape_setter(ndarray_val):

    d = Document()
    with pytest.warns(UserWarning, match='embedding should be a vector') as record:
        d.embedding = ndarray_val

    assert len(record) == 1
