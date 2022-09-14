import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.math import ndarray
import operator


def test_customize_metric_fn():
    N, D = 4, 128
    da = DocumentArray.empty(N)
    da.embeddings = np.random.random([N, D])

    q = np.random.random([D])
    _, r1 = da.find(q)[:, ['scores__cosine__value', 'id']]

    from docarray.math.distance.numpy import cosine

    def inv_cosine(*args):
        return -cosine(*args)

    _, r2 = da.find(q, metric=inv_cosine)[:, ['scores__inv_cosine__value', 'id']]
    assert list(reversed(r1)) == r2


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 32}),
        ('annlite', {'n_dim': 32}),
        ('qdrant', {'n_dim': 32}),
        ('elasticsearch', {'n_dim': 32}),
        ('redis', {'n_dim': 32, 'flush': True}),
    ],
)
@pytest.mark.parametrize('limit', [1, 5, 10])
@pytest.mark.parametrize(
    'query',
    [np.random.random(32), np.random.random((1, 32)), np.random.random((2, 32))],
)
def test_find(storage, config, limit, query, start_storage):
    embeddings = np.random.random((20, 32))

    if config:
        da = DocumentArray(storage=storage, config=config)
    else:
        da = DocumentArray(storage=storage)

    da.extend([Document(embedding=v) for v in embeddings])

    result = da.find(query, limit=limit)
    n_rows_query, n_dim = ndarray.get_array_rows(query)

    if n_rows_query == 1 and n_dim == 1:
        # we expect a result to be DocumentArray
        assert len(result) == limit
    elif n_rows_query == 1 and n_dim == 2:
        # we expect a result to be a list with 1 DocumentArray
        assert len(result) == 1
        assert len(result[0]) == limit
    else:
        # check for each row on the query a DocumentArray is returned
        assert len(result) == n_rows_query

    # check returned objects are sorted according to the storage backend metric
    # weaviate uses cosine similarity by default
    # annlite uses cosine distance by default
    if n_dim == 1:
        if storage == 'weaviate':
            cosine_similarities = [
                t['cosine_similarity'].value for t in result[:, 'scores']
            ]
            assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        if storage == 'redis':
            cosine_distances = [t['score'].value for t in da[:, 'scores']]
            assert sorted(cosine_distances, reverse=False) == cosine_distances
        elif storage in ['memory', 'annlite', 'elasticsearch']:
            cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
            assert sorted(cosine_distances, reverse=False) == cosine_distances
    else:
        if storage == 'weaviate':
            for da in result:
                cosine_similarities = [
                    t['cosine_similarity'].value for t in da[:, 'scores']
                ]
                assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        if storage == 'redis':
            for da in result:
                cosine_distances = [t['score'].value for t in da[:, 'scores']]
                assert sorted(cosine_distances, reverse=False) == cosine_distances
        elif storage in ['memory', 'annlite', 'elasticsearch']:
            for da in result:
                cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
                assert sorted(cosine_distances, reverse=False) == cosine_distances


@pytest.mark.parametrize(
    'storage, config',
    [
        ('elasticsearch', {'n_dim': 32, 'index_text': True}),
        ('redis', {'n_dim': 32, 'flush': True, 'index_text': True}),
    ],
)
def test_find_by_text(storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)
    da.extend(
        [
            Document(id='1', text='token1 token2 token3'),
            Document(id='2', text='token1 token2'),
            Document(id='3', text='token2 token3 token4'),
        ]
    )

    if storage == 'redis':
        results = da.find('token1', scorer='TFIDF')
    else:
        results = da.find('token1')
    assert isinstance(results, DocumentArray)
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '2'}

    results = da.find('token2 token3')
    assert isinstance(results, DocumentArray)
    assert len(results) == 3
    assert set(results[:, 'id']) == {'1', '2', '3'}

    results = da.find('token3 token4')
    assert isinstance(results, DocumentArray)
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}
    results = da.find('token3 token4', limit=1)
    assert len(results) == 1

    results = da.find(['token4', 'token'])
    assert isinstance(results, list)
    assert len(results) == 2  # len(input) = len(output)
    assert len(results[0]) == 1  # 'token4' only appears in one doc
    assert results[0][0].id == '3'  # 'token4' only appears in doc3
    assert len(results[1]) == 0  # 'token' is not present in da vocabulary


@pytest.mark.parametrize(
    'storage, config',
    [
        ('elasticsearch', {'n_dim': 32, 'tag_indices': ['attr1', 'attr2', 'attr3']}),
        (
            'redis',
            {'n_dim': 32, 'flush': True, 'tag_indices': ['attr1', 'attr2', 'attr3']},
        ),
    ],
)
def test_find_by_tag(storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)
    da.extend(
        [
            Document(
                id='1',
                tags={
                    'attr1': 'token1 token2 token3',
                    'attr2': 'token2 token3 token4',
                    'attr3': 'token4 token5 token6',
                },
            ),
            Document(
                id='2',
                tags={
                    'attr1': 'token1',
                    'attr2': 'token2',
                    'attr3': 'token6',
                },
            ),
            Document(
                id='3',
                tags={
                    'attr1': 'token4',
                    'attr2': 'token3',
                    'attr3': 'token1 token5',
                },
            ),
            Document(id='4'),
        ]
    )

    results = da.find('token1 token2', index='attr1')
    assert isinstance(results, DocumentArray)
    assert len(results) == 2
    assert results[0].id == '1'
    assert results[1].id == '2'

    results = da.find('token1 token2', index='attr1', limit=1)
    assert len(results) == 1

    results = da.find('token2 token4', index='attr1')
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}

    results = da.find('token4', index='attr2')
    assert len(results) == 1
    assert results[0].id == '1'

    results = da.find('token6', index='attr3')
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '2'}

    results = da.find('token6', index='attr3', limit=1)
    assert len(results) == 1

    results = da.find('token5', index='attr3')
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}
    assert all(['token5' in r.tags['attr3'] for r in results]) == True

    results = da.find('token1', index='attr3')
    assert len(results) == 1
    assert results[0].id == '3'
    assert all(['token1' in r.tags['attr3'] for r in results]) == True

    results = da.find(['token1 token2'], index='attr1')
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], DocumentArray)

    results = da.find(['token1 token2', 'token1'], index='attr1')
    assert isinstance(results, list)
    assert len(results) == 2
    assert all([isinstance(result, DocumentArray) for result in results]) == True


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


numeric_operators_elasticsearch = {
    'gte': operator.ge,
    'gt': operator.gt,
    'lte': operator.le,
    'lt': operator.lt,
    'eq': operator.eq,
}

numeric_operators_redis = {
    '$gte': operator.ge,
    '$gt': operator.gt,
    '$lte': operator.le,
    '$lt': operator.lt,
    '$eq': operator.eq,
    '$ne': operator.ne,
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
            tuple(
                [
                    'elasticsearch',
                    lambda operator, threshold: {
                        'range': {
                            'price': {
                                operator: threshold,
                            }
                        }
                    },
                    numeric_operators_elasticsearch,
                    operator,
                ]
            )
            for operator in ['gt', 'gte', 'lt', 'lte']
        ],
        *[
            tuple(
                [
                    'redis',
                    lambda operator, threshold: {'price': {operator: threshold}},
                    numeric_operators_redis,
                    operator,
                ]
            )
            for operator in numeric_operators_redis.keys()
        ],
    ],
)
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_search_pre_filtering(
    storage, filter_gen, operator, numeric_operators, start_storage, columns
):
    np.random.seed(0)
    n_dim = 128

    if storage == 'redis':
        da = DocumentArray(
            storage=storage,
            config={'n_dim': n_dim, 'columns': columns, 'flush': True},
        )
    else:
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

        results = da.find(np.random.rand(n_dim), filter=filter)

        assert len(results) > 0

        assert all(
            [numeric_operators[operator](r.tags['price'], threshold) for r in results]
        )


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
                        'valueNumber': threshold,
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
                    'elasticsearch',
                    lambda operator, threshold: {'match': {'price': threshold}},
                    numeric_operators_elasticsearch,
                    operator,
                ]
            )
            for operator in ['eq']
        ],
        *[
            tuple(
                [
                    'elasticsearch',
                    lambda operator, threshold: {
                        'range': {
                            'price': {
                                operator: threshold,
                            }
                        }
                    },
                    numeric_operators_elasticsearch,
                    operator,
                ]
            )
            for operator in ['gt', 'gte', 'lt', 'lte']
        ],
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
            tuple(
                [
                    'redis',
                    lambda operator, threshold: {'price': {operator: threshold}},
                    numeric_operators_redis,
                    operator,
                ]
            )
            for operator in numeric_operators_redis.keys()
        ],
    ],
)
@pytest.mark.parametrize('columns', [[('price', 'float')], {'price': 'float'}])
def test_filtering(
    storage, filter_gen, operator, numeric_operators, start_storage, columns
):
    n_dim = 128

    if storage == 'redis':
        da = DocumentArray(
            storage=storage,
            config={'n_dim': n_dim, 'columns': columns, 'flush': True},
        )
    else:
        da = DocumentArray(storage=storage, config={'n_dim': n_dim, 'columns': columns})

    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(50)])
    thresholds = [10, 20, 30]

    for threshold in thresholds:

        filter = filter_gen(operator, threshold)
        results = da.find(filter=filter)

        assert len(results) > 0

        assert all(
            [numeric_operators[operator](r.tags['price'], threshold) for r in results]
        )


@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_weaviate_filter_query(start_storage, columns):
    n_dim = 128
    da = DocumentArray(storage='weaviate', config={'n_dim': n_dim, 'columns': columns})

    da.extend(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )

    with pytest.raises(ValueError):
        da.find(np.random.rand(n_dim), filter={'wrong': 'filter'})

    with pytest.raises(ValueError):
        da._filter(filter={'wrong': 'filter'})

    assert isinstance(da._filter(filter={}), type(da))


@pytest.mark.parametrize(
    'columns',
    [
        [('price', 'int'), ('category', 'str'), ('size', 'int'), ('isfake', 'bool')],
        {'price': 'int', 'category': 'str', 'size': 'int', 'isfake': 'bool'},
    ],
)
@pytest.mark.parametrize(
    'filter,checker',
    [
        (
            {
                "$or": {
                    "price": {"$gt": 8},
                    "category": {"$eq": "Shoes"},
                },
            },
            lambda r: r.tags['price'] > 8 or r.tags['category'] == 'Shoes',
        ),
        (
            {
                "$and": {
                    "price": {"$ne": 8},
                    "isfake": {"$eq": True},
                },
            },
            lambda r: r.tags['price'] != 8 and r.tags['isfake'] == True,
        ),
        (
            {
                "$or": {
                    "price": {"$lt": 8},
                    "isfake": {"$ne": True},
                },
                "size": {"$lte": 3},
            },
            lambda r: (r.tags['price'] < 8 or r.tags['isfake'] != True)
            and r.tags['size'] <= 3,
        ),
        (
            {
                "$or": {
                    "$and": {
                        "price": {"$gte": 8},
                        "category": {"$ne": "Shoes"},
                    },
                    "size": {"$eq": 3},
                },
            },
            lambda r: (r.tags['price'] >= 8 and r.tags['category'] != 'Shoes')
            or r.tags['size'] == 3,
        ),
    ],
)
def test_redis_category_filter(filter, checker, start_storage, columns):
    n_dim = 128
    da = DocumentArray(
        storage='redis',
        config={
            'n_dim': n_dim,
            'columns': columns,
            'flush': True,
        },
    )

    da.extend(
        [
            Document(
                id=f'r{i}',
                embedding=np.random.rand(n_dim),
                tags={'price': i, 'category': 'Shoes', 'size': i, 'isfake': True},
            )
            for i in range(10)
        ]
    )

    da.extend(
        [
            Document(
                id=f'r{i+10}',
                embedding=np.random.rand(n_dim),
                tags={
                    'price': i,
                    'category': 'Jeans',
                    'size': i,
                    'isfake': False,
                },
            )
            for i in range(10)
        ]
    )

    results = da.find(np.random.rand(n_dim), filter=filter)
    assert len(results) > 0
    assert all([checker(r) for r in results])


@pytest.mark.parametrize('storage', ['memory'])
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_unsupported_pre_filtering(storage, start_storage, columns):

    n_dim = 128
    da = DocumentArray(storage=storage, config={'n_dim': n_dim, 'columns': columns})

    da.extend(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )

    with pytest.raises(ValueError):
        da.find(np.random.rand(n_dim), filter={'price': {'$gte': 2}})


@pytest.mark.parametrize(
    'storage, config',
    [
        ('elasticsearch', {'n_dim': 32, 'index_text': False}),
    ],
)
@pytest.mark.parametrize('limit', [1, 5, 10])
def test_elastic_id_filter(storage, config, limit):
    da = DocumentArray(storage=storage, config=config)
    da.extend([Document(id=f'{i}', embedding=np.random.rand(32)) for i in range(50)])
    id_list = [np.random.choice(50, 10, replace=False) for _ in range(3)]

    for id in id_list:
        id = list(map(lambda x: str(x), id))
        query = {
            "bool": {"filter": {"ids": {"values": id}}},
        }
        result = da.find(query=query, limit=limit)
        assert all([r.id in id for r in result])
        assert len(result) == limit


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2', 'flush': True}),
    ],
)
def test_find_subindex(storage, config):
    n_dim = 3
    subindex_configs = {'@c': None}
    if storage == 'sqlite':
        subindex_configs['@c'] = dict()
    elif storage in ['weaviate', 'annlite', 'qdrant', 'elasticsearch', 'redis']:
        subindex_configs['@c'] = {'n_dim': 2}

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

    if storage in ['sqlite', 'memory']:
        closest_docs = da.find(query=np.array([3, 3]), on='@c', metric='euclidean')
    else:
        closest_docs = da.find(query=np.array([3, 3]), on='@c')

    b = closest_docs[0].embedding == [2, 2]
    if isinstance(b, bool):
        assert b
    else:
        assert b.all()
    for d in closest_docs:
        assert d.id.endswith('_0') or d.id.endswith('_1')


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2', 'flush': True}),
    ],
)
def test_find_subindex_multimodal(storage, config):
    from docarray import dataclass
    from docarray.typing import Text

    @dataclass
    class MMDoc:
        my_text: Text
        my_other_text: Text
        my_third_text: Text

    n_dim = 3
    subindex_configs = {
        '@.[my_text, my_other_text]': {'n_dim': 2},
        '@.[my_third_text]': {'n_dim': 2},
    }

    if storage in ['sqlite', 'memory']:
        subindex_configs['@.[my_text, my_other_text]'] = dict()
        subindex_configs['@.[my_third_text]'] = dict()

    da = DocumentArray(
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    num_docs = 3
    docs_to_add = DocumentArray(
        [
            Document(
                MMDoc(
                    my_text='hello', my_other_text='world', my_third_text='hello again'
                )
            )
            for _ in range(num_docs)
        ]
    )
    for i, d in enumerate(docs_to_add):
        d.id = str(i)
        d.embedding = i * np.ones(n_dim)
        d.my_text.id = str(i) + '_0'
        d.my_text.embedding = np.array([i, i])
        d.my_other_text.id = str(i) + '_1'
        d.my_other_text.embedding = np.array([i, i])
        d.my_third_text.id = str(i) + '_2'
        d.my_third_text.embedding = np.array([3 * i, 3 * i])

    with da:
        da.extend(docs_to_add)

    if storage in ['sqlite', 'memory']:
        closest_docs = da.find(
            query=np.array([3, 3]), on='@.[my_text, my_other_text]', metric='euclidean'
        )
    else:
        closest_docs = da.find(query=np.array([3, 3]), on='@.[my_text, my_other_text]')
    assert (closest_docs[0].embedding == np.array([2, 2])).all()
    for d in closest_docs:
        assert d.id.endswith('_0') or d.id.endswith('_1')

    if storage in ['sqlite', 'memory']:
        closest_docs = da.find(
            query=np.array([3, 3]), on='@.[my_third_text]', metric='euclidean'
        )
    else:
        closest_docs = da.find(query=np.array([3, 3]), on='@.[my_third_text]')
    assert (closest_docs[0].embedding == np.array([3, 3])).all()
    for d in closest_docs:
        assert d.id.endswith('_2')
