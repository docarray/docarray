import numpy as np
import pytest
import scipy.sparse
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig
from tests import random_docs

rand_array = np.random.random([10, 3])


@pytest.fixture()
def docs():
    rand_docs = random_docs(100)
    return rand_docs


@pytest.fixture()
def nested_docs():
    docs = [
        Document(id='r1', chunks=[Document(id='c1'), Document(id='c2')]),
        Document(id='r2', matches=[Document(id='m1'), Document(id='m2')]),
    ]
    return docs


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 3}),
        ('weaviate', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('elasticsearch', {'n_dim': 3}),
        ('redis', {'n_dim': 3}),
        ('milvus', {'n_dim': 3}),
    ],
)
@pytest.mark.parametrize(
    'array',
    [
        rand_array,
        torch.Tensor(rand_array),
        tf.constant(rand_array),
        csr_matrix(rand_array),
    ],
)
def test_set_embeddings_multi_kind(array, storage, config, start_storage):
    da = DocumentArray([Document() for _ in range(10)], storage=storage, config=config)
    da[:, 'embedding'] = array


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_da_get_embeddings(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    with da:
        np.testing.assert_almost_equal(da._get_attributes('embedding'), da.embeddings)
        np.testing.assert_almost_equal(da[:, 'embedding'], da.embeddings)


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_embeddings_setter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    emb = np.random.random((100, 10))
    da[:, 'embedding'] = emb
    with da:
        np.testing.assert_almost_equal(da.embeddings, emb)

    for x, doc in zip(emb, da):
        np.testing.assert_almost_equal(x, doc.embedding)

    da[:, 'embedding'] = None
    if hasattr(da, 'flush'):
        da.flush()
    with da:
        assert da.embeddings is None or not np.any(da.embeddings)


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_embeddings_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    embeddings = np.ones((2, 10))

    with pytest.raises(ValueError):
        with da:
            da.embeddings = embeddings


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_tensors_getter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    tensors = np.random.random((100, 10, 10))
    with da:  # speed up milvus by loading collection
        da.tensors = tensors
        assert len(da) == 100
        np.testing.assert_almost_equal(da.tensors, tensors)

        da.tensors = None
        assert da.tensors is None


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_texts_getter_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    with da:  # speed up milvus by loading collection
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
        assert set(da.texts) == set([''])


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_setter_by_sequences_in_selected_docs_da(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    da[[0, 1, 2], 'text'] = 'test'
    assert da[[0, 1, 2], 'text'] == ['test', 'test', 'test']

    da[[3, 4], 'text'] = ['test', 'test']
    assert da[[3, 4], 'text'] == ['test', 'test']

    da[[0], 'text'] = ['jina']
    assert da[[0], 'text'] == ['jina']

    da[[6], 'text'] = ['test']
    assert da[[6], 'text'] == ['test']

    # test that ID not present in da works
    da[[0], 'id'] = '999'
    assert ['999'] == da[[0], 'id']

    da[[0, 1], 'id'] = ['101', '102']
    assert ['101', '102'] == da[[0, 1], 'id']


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_texts_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    texts = ['hello']

    with pytest.raises(ValueError):
        with da:
            da.texts = texts


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_tensors_wrong_len(docs, config, da_cls, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    tensors = np.ones((2, 10, 10))

    with pytest.raises(ValueError):
        with da:  # speed up milvus by loading collection
            da.tensors = tensors


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_blobs_getter_setter(docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(docs)
    with da:  # speed up milvus by loading collection
        with pytest.raises(ValueError):
            da.blobs = [b'cc', b'bb', b'aa', b'dd']

        da.blobs = [b'aa'] * len(da)
        assert da.blobs == [b'aa'] * len(da)

        da.blobs = None
        if hasattr(da, 'flush'):
            da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert set(da.blobs) == set([b''])


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_ellipsis_getter(nested_docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(nested_docs)
    flattened = da[...]
    assert len(flattened) == 6
    for d, doc_id in zip(flattened, ['c1', 'c2', 'r1', 'm1', 'm2', 'r2']):
        assert d.id == doc_id


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_ellipsis_attribute_setter(nested_docs, da_cls, config, start_storage):
    if config:
        da = da_cls(config=config)
    else:
        da = da_cls()
    da.extend(nested_docs)
    da[..., 'text'] = 'new'
    assert all(d.text == 'new' for d in da[...])


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=6)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=6)),
        (DocumentArrayElastic, ElasticConfig(n_dim=6)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=6)),
    ],
)
def test_zero_embeddings(da_cls, config, start_storage):
    a = np.zeros([10, 6])
    if config:
        da = da_cls.empty(10, config=config)
    else:
        da = da_cls.empty(10)

    with da:  # speed up milvus by loading collection
        # all zero, dense
        da[:, 'embedding'] = a
        np.testing.assert_almost_equal(da.embeddings, a)
        for d in da:
            assert d.embedding.shape == (6,)

        # all zero, sparse
        sp_a = scipy.sparse.coo_matrix(a)
        da[:, 'embedding'] = sp_a
        np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
        for d in da:
            # scipy sparse row-vector can only be a (1, m) not squeezible
            assert d.embedding.shape == (1, 6)

        # near zero, sparse
        a = np.random.random([10, 6])
        a[a > 0.1] = 0
        sp_a = scipy.sparse.coo_matrix(a)
        da[:, 'embedding'] = sp_a
        np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
        for d in da:
            # scipy sparse row-vector can only be a (1, m) not squeezible
            assert d.embedding.shape == (1, 6)


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
def test_getset_subindex(storage, config):

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
    with da:
        da[0] = Document(
            embedding=-1 * np.ones(n_dim),
            chunks=[
                Document(id='c_0', embedding=np.array([-1, -1])),
                Document(id='c_1', embedding=np.array([-2, -2])),
            ],
        )

    with da:
        da[1:] = [
            Document(
                embedding=-1 * np.ones(n_dim),
                chunks=[
                    Document(id='c_0' + str(i), embedding=np.array([-1, -1])),
                    Document(id='c_1' + str(i), embedding=np.array([-2, -2])),
                ],
            )
            for i in range(2)
        ]

    # test insert single doc
    assert embeddings_eq(da[0].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[0].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[0].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_0'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_1'].embedding, [-2, -2])

    # test insert slice of docs
    assert embeddings_eq(da[1].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[1].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[1].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_00'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_10'].embedding, [-2, -2])

    assert embeddings_eq(da[2].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[2].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[2].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_01'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_11'].embedding, [-2, -2])


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
def test_init_subindex(storage, config):
    num_top_level_docs = 5
    num_chunks_per_doc = 3
    subindex_configs = (
        {'@c': None} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        [
            Document(
                chunks=[Document(text=f'{i}{j}') for j in range(num_chunks_per_doc)]
            )
            for i in range(num_top_level_docs)
        ],
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    assert len(da['@c']) == num_top_level_docs * num_chunks_per_doc
    assert len(da._subindices['@c']) == num_top_level_docs * num_chunks_per_doc
    expected_texts = []
    for i in range(num_top_level_docs):
        for j in range(num_chunks_per_doc):
            expected_texts.append(f'{i}{j}')
    assert da['@c'].texts == expected_texts
    assert da._subindices['@c'].texts == expected_texts


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
def test_set_on_subindex(storage, config):
    n_dim = 3
    subindex_configs = (
        {'@c': dict()} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        [Document(chunks=[Document() for j in range(3)]) for i in range(5)],
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    embeddings_to_assign = np.random.random((5 * 3, 2))
    with da:
        da['@c'].embeddings = embeddings_to_assign
    with da:
        assert (da['@c'].embeddings == embeddings_to_assign).all()
        assert (da._subindices['@c'].embeddings == embeddings_to_assign).all()

    with da:
        da['@c'].texts = ['hello' for _ in range(5 * 3)]
    with da:
        assert da['@c'].texts == ['hello' for _ in range(5 * 3)]
        assert da._subindices['@c'].texts == ['hello' for _ in range(5 * 3)]

    matches = da.find(query=np.random.random(2), on='@c')
    assert matches
    assert len(matches[0].embedding) == 2


def test_raise_correct_error_subindex_set():
    da = DocumentArray(
        [
            Document(chunks=[Document(text='hello')]),
            Document(chunks=[Document(text='world')]),
        ],
        subindex_configs={'@c': None},
    )

    with pytest.raises(ValueError):
        da['@c'] = DocumentArray(Document() for _ in range(2))
