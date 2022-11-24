import copy

import numpy as np
import paddle
import pytest
import scipy.sparse as sp

import tensorflow as tf
import torch
from scipy.sparse import csr_matrix, bsr_matrix, coo_matrix, csc_matrix
from scipy.spatial.distance import cdist as scipy_cdist

from docarray import Document, DocumentArray
import operator


@pytest.fixture()
def doc_lists():
    d1 = Document(embedding=np.array([0, 0, 0]))
    d2 = Document(embedding=np.array([3, 0, 0]))
    d3 = Document(embedding=np.array([1, 0, 0]))
    d4 = Document(embedding=np.array([2, 0, 0]))

    d1_m = Document(embedding=np.array([1, 0, 0]))
    d2_m = Document(embedding=np.array([2, 0, 0]))
    d3_m = Document(embedding=np.array([0, 0, 1]))
    d4_m = Document(embedding=np.array([0, 0, 2]))
    d5_m = Document(embedding=np.array([0, 0, 3]))

    return [d1, d2, d3, d4], [d1_m, d2_m, d3_m, d4_m, d5_m]


@pytest.fixture
def docarrays_for_embedding_distance_computation(doc_lists):
    D1, D2 = doc_lists
    da1 = DocumentArray(D1)
    da2 = DocumentArray(D2)
    return da1, da2


@pytest.fixture
def docarrays_for_embedding_distance_computation_sparse():
    d1 = Document(embedding=sp.csr_matrix([0, 0, 0]))
    d2 = Document(embedding=sp.csr_matrix([3, 0, 0]))
    d3 = Document(embedding=sp.csr_matrix([1, 0, 0]))
    d4 = Document(embedding=sp.csr_matrix([2, 0, 0]))

    d1_m = Document(embedding=sp.csr_matrix([1, 0, 0]))
    d2_m = Document(embedding=sp.csr_matrix([2, 0, 0]))
    d3_m = Document(embedding=sp.csr_matrix([0, 0, 1]))
    d4_m = Document(embedding=sp.csr_matrix([0, 0, 2]))
    d5_m = Document(embedding=sp.csr_matrix([0, 0, 3]))

    D1 = DocumentArray([d1, d2, d3, d4])
    D2 = DocumentArray([d1_m, d2_m, d3_m, d4_m, d5_m])
    return D1, D2


@pytest.fixture
def embeddings():
    return np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])


def doc_lists_to_doc_arrays(doc_lists, *args, **kwargs):
    doc_list1, doc_list2 = doc_lists
    D1 = DocumentArray()
    D1.extend(doc_list1)
    D2 = DocumentArray()
    D2.extend(doc_list2)
    return D1, D2


@pytest.mark.parametrize(
    'storage, config',
    [
        ('annlite', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('weaviate', {'n_dim': 3}),
        ('redis', {'n_dim': 3}),
        ('milvus', {'n_dim': 3}),
    ],
)
@pytest.mark.parametrize('limit', [1, 2, 3])
@pytest.mark.parametrize('exclude_self', [True, False])
def test_match(storage, config, doc_lists, limit, exclude_self, start_storage):
    D1, D2 = doc_lists_to_doc_arrays(doc_lists)

    if config:
        da = DocumentArray(D2, storage=storage, config=config)
    else:
        da = DocumentArray(D2, storage=storage)

    D1.match(da, limit=limit, exclude_self=exclude_self)
    for m in D1[:, 'matches']:
        assert len(m) == limit

    if storage == 'redis':
        expected_sorted_values = [
            D1[0].matches[i].scores['score'].value for i in range(limit)
        ]
    else:
        expected_sorted_values = [
            D1[0].matches[i].scores['cosine'].value for i in range(limit)
        ]

    assert expected_sorted_values == sorted(expected_sorted_values)


@pytest.mark.parametrize(
    'limit, batch_size', [(1, None), (2, None), (None, None), (1, 1), (1, 2), (2, 1)]
)
@pytest.mark.parametrize('only_id', [True, False])
def test_matching_retrieves_correct_number(
    doc_lists,
    limit,
    batch_size,
    tmpdir,
    only_id,
):
    D1, D2 = doc_lists_to_doc_arrays(
        doc_lists,
    )

    D1.match(
        D2, metric='sqeuclidean', limit=limit, batch_size=batch_size, only_id=only_id
    )
    for m in D1[:, 'matches']:
        if limit is None:
            assert len(m) == len(D2)
        else:
            assert len(m) == limit


@pytest.mark.parametrize('metric', ['sqeuclidean', 'cosine'])
@pytest.mark.parametrize('only_id', [True, False])
def test_matching_same_results_with_sparse(
    docarrays_for_embedding_distance_computation,
    docarrays_for_embedding_distance_computation_sparse,
    metric,
    only_id,
):
    D1, D2 = docarrays_for_embedding_distance_computation
    D1_sp, D2_sp = docarrays_for_embedding_distance_computation_sparse

    # use match with numpy arrays
    D1.match(D2, metric=metric, only_id=only_id)
    distances = []
    for m in D1[:, 'matches']:
        for d in m:
            distances.extend([d.scores[metric].value])

    # use match with sparse arrays
    D1_sp.match(D2_sp, metric=metric, is_sparse=True)
    distances_sparse = []
    for m in D1[:, 'matches']:
        for d in m:
            distances_sparse.extend([d.scores[metric].value])

    np.testing.assert_equal(distances, distances_sparse)


@pytest.mark.parametrize('metric', ['sqeuclidean', 'cosine'])
@pytest.mark.parametrize('only_id', [True, False])
def test_matching_same_results_with_batch(
    docarrays_for_embedding_distance_computation, metric, only_id
):
    D1, D2 = docarrays_for_embedding_distance_computation
    D1_batch = copy.deepcopy(D1)
    D2_batch = copy.deepcopy(D2)

    # use match without batches
    D1.match(D2, metric=metric, only_id=only_id)
    distances = []
    for m in D1[:, 'matches']:
        for d in m:
            distances.extend([d.scores[metric].value])

    # use match with batches
    D1_batch.match(D2_batch, metric=metric, batch_size=10)

    distances_batch = []
    for m in D1[:, 'matches']:
        for d in m:
            distances_batch.extend([d.scores[metric].value])

    np.testing.assert_equal(distances, distances_batch)


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('only_id', [True, False])
def test_matching_scipy_cdist(
    docarrays_for_embedding_distance_computation, metric, only_id
):
    def scipy_cdist_metric(X, Y, *args):
        return scipy_cdist(X, Y, metric=metric)

    D1, D2 = docarrays_for_embedding_distance_computation
    D1_scipy = copy.deepcopy(D1)

    # match with our custom metric
    D1.match(D2, metric=metric)
    distances = []
    for m in D1[:, 'matches']:
        for d in m:
            distances.extend([d.scores[metric].value])

    # match with callable cdist function from scipy
    D1_scipy.match(D2, metric=scipy_cdist_metric, only_id=only_id)
    distances_scipy = []
    for m in D1[:, 'matches']:
        for d in m:
            distances_scipy.extend([d.scores[metric].value])

    np.testing.assert_equal(distances, distances_scipy)


@pytest.mark.parametrize(
    'normalization, metric',
    [
        ((0, 1), 'sqeuclidean'),
        (None, 'euclidean'),
        ((0, 1), 'euclidean'),
        (None, 'cosine'),
        ((0, 1), 'cosine'),
    ],
)
@pytest.mark.parametrize('use_scipy', [True, False])
@pytest.mark.parametrize('only_id', [True, False])
def test_matching_retrieves_closest_matches(
    doc_lists,
    normalization,
    metric,
    use_scipy,
    only_id,
):
    """
    Tests if match.values are returned 'low to high' if normalization is True or 'high to low' otherwise
    """
    D1, D2 = doc_lists_to_doc_arrays(
        doc_lists,
    )
    D1.match(
        D2,
        metric=metric,
        limit=3,
        normalization=normalization,
        use_scipy=use_scipy,
        only_id=only_id,
    )
    expected_sorted_values = [
        D1[0].matches[i].scores['sqeuclidean'].value for i in range(3)
    ]
    if normalization:
        assert min(expected_sorted_values) >= 0
        assert max(expected_sorted_values) <= 1
    else:
        assert expected_sorted_values == sorted(expected_sorted_values)


@pytest.mark.parametrize('blob_pool_size', [1000, 3])
@pytest.mark.parametrize('first_memmap', [True, False])
@pytest.mark.parametrize('second_memmap', [True, False])
@pytest.mark.parametrize('only_id', [True, False])
def test_2arity_function(
    first_memmap, second_memmap, doc_lists, tmpdir, blob_pool_size, only_id
):
    def dotp(x, y, *args):
        return np.dot(x, np.transpose(y))

    D1, D2 = doc_lists_to_doc_arrays(
        doc_lists,
        tmpdir,
        first_memmap,
        second_memmap,
        blob_pool_size=blob_pool_size,
    )
    D1.match(D2, metric=dotp, use_scipy=True, only_id=only_id)

    for d in D1:
        for m in d.matches:
            assert 'dotp' in m.scores


@pytest.mark.parametrize('only_id', [True, False])
def test_match_inclusive(only_id):
    """Call match function, while the other :class:`DocumentArray` is itself
    or have same :class:`Document`.
    """
    # The document array da1 match with itself.
    da1 = DocumentArray(
        [
            Document(embedding=np.array([1, 2, 3])),
            Document(embedding=np.array([1, 0, 1])),
            Document(embedding=np.array([1, 1, 2])),
        ]
    )

    da1.match(da1, only_id=only_id)
    assert len(da1) == 3
    traversed = da1.traverse_flat(traversal_paths='m,mm,mmm')
    assert len(traversed) == 9
    # The document array da2 shares same documents with da1
    da2 = DocumentArray([Document(embedding=np.array([4, 1, 3])), da1[0], da1[1]])
    da1.match(da2, only_id=only_id)
    assert len(da2) == 3
    traversed = da1.traverse_flat(traversal_paths='m,mm,mmm')
    assert len(traversed) == 9


@pytest.mark.parametrize('exclude_self, num_matches', [(True, 1), (False, 2)])
@pytest.mark.parametrize('only_id', [True, False])
def test_match_exclude_self(exclude_self, num_matches, only_id):
    da1 = DocumentArray(
        [
            Document(id='1', embedding=np.array([1, 2])),
            Document(id='2', embedding=np.array([3, 4])),
        ]
    )
    da2 = DocumentArray(
        [
            Document(id='1', embedding=np.array([1, 2])),
            Document(id='2', embedding=np.array([3, 4])),
        ]
    )
    da1.match(da2, exclude_self=exclude_self, only_id=only_id)
    for d in da1:
        assert len(d.matches) == num_matches


@pytest.fixture()
def get_pair_document_array():
    da1 = DocumentArray(
        [
            Document(id='1', embedding=np.array([1, 2])),
            Document(id='2', embedding=np.array([3, 4])),
        ]
    )
    da2 = DocumentArray(
        [
            Document(id='1', embedding=np.array([1, 2])),
            Document(id='2', embedding=np.array([3, 4])),
            Document(id='3', embedding=np.array([4, 5])),
        ]
    )
    yield da1, da2


@pytest.mark.parametrize(
    'limit, expect_len, exclude_self',
    [
        (2, 2, True),
        (1, 1, True),
        (3, 2, True),
        (2, 2, False),
        (1, 1, False),
        (3, 3, False),
    ],
)
def test_match_exclude_self_limit_2(
    get_pair_document_array, exclude_self, limit, expect_len
):
    da1, da2 = get_pair_document_array
    da1.match(da2, exclude_self=exclude_self, limit=limit)
    for d in da1:
        assert len(d.matches) == expect_len


@pytest.mark.parametrize(
    'lhs, rhs',
    [
        (DocumentArray(), DocumentArray()),
        (
            DocumentArray(
                [
                    Document(embedding=np.array([3, 4])),
                    Document(embedding=np.array([4, 5])),
                ]
            ),
            DocumentArray(
                [
                    Document(embedding=np.array([3, 4])),
                    Document(embedding=np.array([4, 5])),
                ]
            ),
        ),
        (
            DocumentArray(),
            DocumentArray(
                [
                    Document(embedding=np.array([3, 4])),
                    Document(embedding=np.array([4, 5])),
                ]
            ),
        ),
        (
            (
                DocumentArray(
                    [
                        Document(embedding=np.array([3, 4])),
                        Document(embedding=np.array([4, 5])),
                    ]
                )
            ),
            DocumentArray(),
        ),
        (None, DocumentArray()),
        (DocumentArray(), None),
    ],
)
def test_match_none(lhs, rhs):
    if lhs is not None:
        lhs.match(rhs)
    if rhs is not None:
        rhs.match(lhs)


@pytest.fixture()
def get_two_docarray():
    d1 = Document(embedding=np.array([0, 0, 0]))
    d1c1 = Document(embedding=np.array([0, 1, 0]))

    d2 = Document(embedding=np.array([1, 0, 0]))
    d2c1 = Document(embedding=np.array([1, 1, 0]))
    d2c2 = Document(embedding=np.array([1, 0, 1]))

    d3 = Document(embedding=np.array([2, 1, 1]))
    d3c1 = Document(embedding=np.array([2, 1, 0]))
    d3c2 = Document(embedding=np.array([2, 0, 1]))
    d3c3 = Document(embedding=np.array([2, 0, 0]))

    d4 = Document(embedding=np.array([3, 1, 1]))
    d4c1 = Document(embedding=np.array([3, 1, 0]))
    d4c2 = Document(embedding=np.array([3, 0, 1]))
    d4c3 = Document(embedding=np.array([3, 0, 0]))
    d4c4 = Document(embedding=np.array([3, 1, 1]))

    d1.chunks.extend([d1c1])
    d2.chunks.extend([d2c1, d2c2])
    d3.chunks.extend([d3c1, d3c2, d3c3])
    d4.chunks.extend([d4c1, d4c2, d4c3, d4c4])

    da1 = DocumentArray([d1, d2])
    da2 = DocumentArray([d3, d4])
    yield da1, da2


def test_match_with_traversal_path(get_two_docarray):
    da1, da2 = get_two_docarray
    da1.match(da2.traverse_flat('c'))
    assert len(da1[0].matches) == len(da2[0].chunks) + len(da2[1].chunks)

    da2.match(da1.traverse_flat('c'))
    assert len(da2[0].matches) == len(da1[0].chunks) + len(da1[1].chunks)


def test_match_on_two_sides_chunks(get_two_docarray):
    da1, da2 = get_two_docarray
    da2.traverse_flat('c').match(da1.traverse_flat('c'))
    assert len(da2[0].matches) == 0
    assert len(da2[0].chunks[0].matches) == len(da1[0].chunks) + len(da1[1].chunks)

    da1.traverse_flat('c').match(da2.traverse_flat('c'))
    assert len(da1[0].matches) == 0
    assert len(da1[0].chunks[0].matches) == len(da2[0].chunks) + len(da2[1].chunks)


@pytest.mark.parametrize('exclude_self', [True, False])
@pytest.mark.parametrize('limit', [1, 2, 3])
def test_exclude_self_should_keep_limit(limit, exclude_self):
    da = DocumentArray(
        [
            Document(embedding=np.array([3, 1, 0])),
            Document(embedding=np.array([3, 0, 1])),
            Document(embedding=np.array([3, 0, 0])),
            Document(embedding=np.array([3, 1, 1])),
        ]
    )
    da.match(da, exclude_self=exclude_self, limit=limit)
    for d in da:
        assert len(d.matches) == limit
        if exclude_self:
            for m in d.matches:
                assert d.id != m.id


@pytest.mark.parametrize('only_id', [True, False])
def test_only_id(docarrays_for_embedding_distance_computation, only_id):
    D1, D2 = docarrays_for_embedding_distance_computation
    D1.match(D2, only_id=only_id)
    for d in D1:
        for m in d.matches:
            assert (m.embedding is None) == only_id
            assert m.id


@pytest.mark.parametrize(
    'match_kwargs',
    [
        dict(limit=5, normalization=(1, 0), batch_size=10),
        dict(normalization=(1, 0), batch_size=10),
        dict(normalization=(1, 0)),
        dict(),
    ],
)
@pytest.mark.parametrize('nnz_ratio', [0.5, 1])
def test_dense_vs_sparse_match(match_kwargs, nnz_ratio):
    N = 100
    D = 256
    sp_embed = np.random.random([N, D])
    sp_embed[sp_embed > nnz_ratio] = 0

    da1 = DocumentArray.empty(N)
    da2 = DocumentArray.empty(N)

    # use sparse embedding
    da1.embeddings = sp.coo_matrix(sp_embed)
    da1.texts = [str(j) for j in range(N)]
    size_sp = sum(d.nbytes for d in da1)
    da1.match(da1, **match_kwargs)

    sparse_result = [m.text for m in da1[0].matches]

    # use dense embedding
    da2.embeddings = sp_embed
    da2.texts = [str(j) for j in range(N)]
    size_dense = sum(d.nbytes for d in da2)
    da2.match(da2, **match_kwargs)
    dense_result = [m.text for m in da2[0].matches]

    assert sparse_result == dense_result

    print(
        f'sparse DA: {size_sp} bytes is {size_sp / size_dense * 100:.0f}% of dense DA {size_dense} bytes'
    )


def get_ndarrays():
    a = np.random.random([10, 3])
    a[a > 0.5] = 0
    return [
        a,
        torch.tensor(a),
        tf.constant(a),
        paddle.to_tensor(a),
        csr_matrix(a),
        bsr_matrix(a),
        coo_matrix(a),
        csc_matrix(a),
    ]


@pytest.mark.parametrize('ndarray_val', get_ndarrays())
def test_diff_framework_match(ndarray_val):
    da = DocumentArray.empty(10)
    da.embeddings = ndarray_val
    da.match(da)


def test_match_ensure_scores_unique():
    import numpy as np
    from docarray import DocumentArray

    da1 = DocumentArray.empty(4)
    da1.embeddings = np.array(
        [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 2, 2, 1, 0]]
    )

    da2 = DocumentArray.empty(5)
    da2.embeddings = np.array(
        [
            [0.0, 0.1, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0, 0.0],
            [1.0, 1.2, 1.0, 1.0, 0.0],
            [1.0, 2.2, 2.0, 1.0, 0.0],
            [4.0, 5.2, 2.0, 1.0, 0.0],
        ]
    )

    da1.match(da2, metric='euclidean', only_id=False, limit=5)

    assert len(da1) == 4
    for query in da1:
        previous_score = -10000
        assert len(query.matches) == 5
        for m in query.matches:
            assert m.scores['euclidean'].value >= previous_score
            previous_score = m.scores['euclidean'].value


numeric_operators_annlite = {
    '$gte': operator.ge,
    '$gt': operator.gt,
    '$lte': operator.le,
    '$lt': operator.lt,
    '$eq': operator.eq,
    '$neq': operator.ne,
}

numeric_operators_weaviate = {
    'GreaterThanEqual': operator.ge,
    'GreaterThan': operator.gt,
    'LessThanEqual': operator.le,
    'LessThan': operator.lt,
    'Equal': operator.eq,
    'NotEqual': operator.ne,
}


numeric_operators_qdrant = {
    'gte': operator.ge,
    'gt': operator.gt,
    'lte': operator.le,
    'lt': operator.lt,
    'eq': operator.eq,
    'neq': operator.ne,
}

numeric_operators_redis = {
    'gte': operator.ge,
    'gt': operator.gt,
    'lte': operator.le,
    'lt': operator.lt,
    'eq': operator.eq,
    'ne': operator.ne,
}


@pytest.mark.parametrize(
    'storage,filter_gen,numeric_operators,operator',
    [
        *[
            tuple(
                [
                    'weaviate',
                    lambda operator, threshold: {
                        'path': ['price'],
                        'operator': operator,
                        'valueInt': threshold,
                    },
                    numeric_operators_weaviate,
                    operator,
                ]
            )
            for operator in numeric_operators_weaviate.keys()
        ],
        *[
            tuple(
                [
                    'qdrant',
                    lambda operator, threshold: {
                        'must': [{'key': 'price', 'range': {operator: threshold}}]
                    },
                    numeric_operators_qdrant,
                    operator,
                ]
            )
            for operator in ['gte', 'gt', 'lte', 'lt']
        ],
        tuple(
            [
                'qdrant',
                lambda operator, threshold: {
                    'must': [{'key': 'price', 'match': {'value': threshold}}]
                },
                numeric_operators_qdrant,
                'eq',
            ]
        ),
        tuple(
            [
                'qdrant',
                lambda operator, threshold: {
                    'must_not': [{'key': 'price', 'match': {'value': threshold}}]
                },
                numeric_operators_qdrant,
                'neq',
            ]
        ),
        *[
            tuple(
                [
                    'annlite',
                    lambda operator, threshold: {'price': {operator: threshold}},
                    numeric_operators_annlite,
                    operator,
                ]
            )
            for operator in numeric_operators_annlite.keys()
        ],
        *[
            (
                'redis',
                lambda operator, threshold: f'@price:[{threshold} inf] ',
                numeric_operators_redis,
                'gte',
            ),
            (
                'redis',
                lambda operator, threshold: f'@price:[({threshold} inf] ',
                numeric_operators_redis,
                'gt',
            ),
            (
                'redis',
                lambda operator, threshold: f'@price:[-inf {threshold}] ',
                numeric_operators_redis,
                'lte',
            ),
            (
                'redis',
                lambda operator, threshold: f'@price:[-inf ({threshold}] ',
                numeric_operators_redis,
                'lt',
            ),
            (
                'redis',
                lambda operator, threshold: f'@price:[{threshold} {threshold}] ',
                numeric_operators_redis,
                'eq',
            ),
            (
                'redis',
                lambda operator, threshold: f'(- @price:[{threshold} {threshold}]) ',
                numeric_operators_redis,
                'ne',
            ),
        ],
    ],
)
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_match_pre_filtering(
    storage, filter_gen, operator, numeric_operators, start_storage, columns
):
    n_dim = 128

    da = DocumentArray(storage=storage, config={'n_dim': n_dim, 'columns': columns})

    da.extend(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    thresholds = [10, 20, 30]

    for threshold in thresholds:

        filter = filter_gen(operator, threshold)

        doc = Document(embedding=np.random.rand(n_dim))
        doc.match(da, filter=filter)

        assert len(doc.matches) > 0

        assert all(
            [
                numeric_operators[operator](r.tags['price'], threshold)
                for r in doc.matches
            ]
        )


def embeddings_eq(emb1, emb2):
    b = emb1 == emb2
    if isinstance(b, bool):
        return b
    else:
        return b.all()


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
    ],
)
def test_match_subindex(storage, config):
    n_dim = 3
    subindex_configs = (
        {'@c': dict()} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(n_dim),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=np.array([i, i])),
                        Document(id=str(i) + '_1', embedding=np.array([i, i])),
                    ],
                )
                for i in range(3)
            ]
        )

    query = Document(embedding=np.array([3, 3]))
    if storage in ['sqlite', 'memory']:
        query.match(da, on='@c', metric='euclidean')
    else:
        query.match(da, on='@c')
    closest_docs = query.matches

    assert embeddings_eq(closest_docs[0].embedding, [2, 2])
    for d in closest_docs:
        assert d.id.endswith('_0') or d.id.endswith('_1')
