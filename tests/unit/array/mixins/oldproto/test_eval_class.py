import copy

import numpy as np
import pytest

from datasets import load_dataset
from string import printable
from collections import Counter

from transformers import BertModel, BertConfig, BertTokenizer

from docarray import DocumentArray, Document


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('r_precision', {}),
        ('precision_at_k', {}),
        ('hit_at_k', {}),
        ('average_precision', {}),
        ('reciprocal_rank', {}),
        ('recall_at_k', {'max_rel': 9}),
        ('f1_score_at_k', {'max_rel': 9}),
        ('ndcg_at_k', {}),
    ],
)
def test_eval_mixin_perfect_match(metric_fn, kwargs, storage, config, start_storage):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 256])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)
    r = da1.evaluate(ground_truth=da1, metrics=[metric_fn], strict=False, **kwargs)[
        metric_fn
    ]
    assert isinstance(r, float)
    assert r == 1.0
    for d in da1:
        assert d.evaluations[metric_fn].value == 1.0


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_eval_mixin_perfect_match_multiple_metrics(storage, config, start_storage):
    metric_fns = [
        'r_precision',
        'precision_at_k',
        'hit_at_k',
        'average_precision',
        'reciprocal_rank',
        'recall_at_k',
        'f1_score_at_k',
        'ndcg_at_k',
    ]
    kwargs = {'max_rel': 9}
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 256])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)
    r = da1.evaluate(ground_truth=da1, metrics=metric_fns, strict=False, **kwargs)
    for metric_fn in metric_fns:
        assert metric_fn in r
        assert isinstance(r[metric_fn], float)
        assert r[metric_fn] == 1.0
        for d in da1:
            assert d.evaluations[metric_fn].value == 1.0


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
    ],
)
@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('r_precision', {}),
        ('precision_at_k', {}),
        ('hit_at_k', {}),
        ('average_precision', {}),
        ('reciprocal_rank', {}),
        ('recall_at_k', {'max_rel': 9}),
        ('f1_score_at_k', {'max_rel': 9}),
        ('ndcg_at_k', {}),
    ],
)
def test_eval_mixin_perfect_match_labeled(
    metric_fn, kwargs, storage, config, start_storage
):
    da1 = DocumentArray.empty(10)
    for d in da1:
        d.tags = {'label': 'A'}
    da1.embeddings = np.random.random([10, 256])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)
    r = da1.evaluate(metrics=[metric_fn], **kwargs)[metric_fn]
    assert isinstance(r, float)
    assert r == 1.0
    for d in da1:
        assert d.evaluations[metric_fn].value == 1.0


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('r_precision', {}),
        ('precision_at_k', {}),
        ('hit_at_k', {}),
        ('average_precision', {}),
        ('reciprocal_rank', {}),
        ('recall_at_k', {'max_rel': 9}),
        ('f1_score_at_k', {'max_rel': 9}),
        ('ndcg_at_k', {}),
    ],
)
def test_eval_mixin_zero_labeled(storage, config, metric_fn, start_storage, kwargs):
    da1 = DocumentArray.empty(10)
    for d in da1:
        d.tags = {'label': 'A'}
    da1.embeddings = np.random.random([10, 256])
    da2 = copy.deepcopy(da1)
    for d in da2:
        d.tags = {'label': 'B'}
    da1_index = DocumentArray(da2, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)
    r = da1.evaluate([metric_fn], **kwargs)[metric_fn]
    assert isinstance(r, float)
    assert r == 0.0
    for d in da1:
        assert d.evaluations[metric_fn].value == 0.0


@pytest.mark.parametrize('label_tag', ['label', 'custom_tag'])
@pytest.mark.parametrize(
    'metric_fn, metric_score',
    [
        ('r_precision', 1.0 / 3),
        ('precision_at_k', 1.0 / 3),
        ('hit_at_k', 1.0),
        ('average_precision', (1.0 + 0.5 + (1.0 / 3)) / 3),
        ('reciprocal_rank', (1.0 + 0.5 + (1.0 / 3)) / 3),
        ('recall_at_k', 1.0 / 3),
        ('f1_score_at_k', 1.0 / 3),
        ('dcg_at_k', (1.0 + 1.0 + 0.6309) / 3),
    ],
)
def test_eval_mixin_one_of_n_labeled(metric_fn, metric_score, label_tag):
    da = DocumentArray([Document(text=str(i), tags={label_tag: i}) for i in range(3)])
    for d in da:
        d.matches = da
    r = da.evaluate([metric_fn], label_tag=label_tag)[metric_fn]
    assert abs(r - metric_score) < 0.001


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('r_precision', {}),
        ('precision_at_k', {}),
        ('hit_at_k', {}),
        ('average_precision', {}),
        ('reciprocal_rank', {}),
        ('recall_at_k', {'max_rel': 9}),
        ('f1_score_at_k', {'max_rel': 9}),
        ('ndcg_at_k', {}),
    ],
)
def test_eval_mixin_zero_match(storage, config, metric_fn, start_storage, kwargs):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 256])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)

    da2 = copy.deepcopy(da1)
    da2.embeddings = np.random.random([10, 256])
    da2_index = DocumentArray(da2, storage=storage, config=config)
    da2.match(da2_index, exclude_self=True)

    r = da1.evaluate(ground_truth=da2, metrics=[metric_fn], **kwargs)[metric_fn]
    assert isinstance(r, float)
    assert r == 1.0
    for d in da1:
        d: Document
        assert d.evaluations[metric_fn].value == 1.0


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_diff_len_should_raise(storage, config, start_storage):
    da1 = DocumentArray.empty(10)
    da2 = DocumentArray.empty(5)
    for d in da2:
        d.matches.append(da2[0])
    da2 = DocumentArray(da2, storage=storage, config=config)
    with pytest.raises(ValueError):
        da1.evaluate(ground_truth=da2, metrics=['precision_at_k'])


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_diff_hash_fun_should_raise(storage, config, start_storage):
    da1 = DocumentArray.empty(10)
    da2 = DocumentArray.empty(5)
    for d in da2:
        d.matches.append(da2[0])
    da2 = DocumentArray(da2, storage=storage, config=config)
    with pytest.raises(ValueError):
        da1.evaluate(ground_truth=da2, metrics=['precision_at_k'])


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('elasticsearch', {'n_dim': 3}),
        ('redis', {'n_dim': 3}),
        ('milvus', {'n_dim': 3}),
    ],
)
def test_same_hash_same_len_fun_should_work(storage, config, start_storage):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 3])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index)
    da2 = DocumentArray.empty(10)
    da2.embeddings = np.random.random([10, 3])
    da2_index = DocumentArray(da1, storage=storage, config=config)
    da2.match(da2_index)
    with pytest.raises(ValueError):
        da1.evaluate(ground_truth=da2, metrics=['precision_at_k'])
    for d1, d2 in zip(da1, da2):
        d1.id = d2.id

    da1.evaluate(ground_truth=da2, metrics=['precision_at_k'])


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('elasticsearch', {'n_dim': 3}),
        ('redis', {'n_dim': 3}),
        ('milvus', {'n_dim': 3}),
    ],
)
def test_adding_noise(storage, config, start_storage):
    da = DocumentArray.empty(10)

    da.embeddings = np.random.random([10, 3])
    da_index = DocumentArray(da, storage=storage, config=config)
    da.match(da_index, exclude_self=True)

    da2 = copy.deepcopy(da)

    for d in da2:
        d.matches.extend(DocumentArray.empty(10))
        d.matches = d.matches.shuffle()

    assert (
        da2.evaluate(ground_truth=da, metrics=['precision_at_k'], k=10)[
            'precision_at_k'
        ]
        < 1.0
    )

    for d in da2:
        assert 0.0 < d.evaluations['precision_at_k'].value < 1.0


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 128}),
        ('qdrant', {'n_dim': 128}),
        ('elasticsearch', {'n_dim': 128}),
        ('redis', {'n_dim': 128}),
        ('milvus', {'n_dim': 128}),
    ],
)
@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('recall_at_k', {}),
        ('f1_score_at_k', {}),
    ],
)
def test_diff_match_len_in_gd(storage, config, metric_fn, start_storage, kwargs):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 128])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1, exclude_self=True)

    da2 = copy.deepcopy(da1)
    da2.embeddings = np.random.random([10, 128])
    da2_index = DocumentArray(da2, storage=storage, config=config)
    da2.match(da2_index, exclude_self=True)
    # pop some matches from first document
    da2[0].matches.pop(8)

    r = da1.evaluate(ground_truth=da2, metrics=[metric_fn], **kwargs)[metric_fn]
    assert isinstance(r, float)
    np.testing.assert_allclose(r, 1.0, rtol=1e-2)  #
    for d in da1:
        d: Document
        # f1_score does not yield 1 for the first document as one of the match is missing
        assert d.evaluations[metric_fn].value > 0.9


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_empty_da_should_raise(storage, config, start_storage):
    da = DocumentArray([], storage=storage, config=config)
    with pytest.raises(ValueError):
        da.evaluate(metrics=['precision_at_k'])


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_missing_groundtruth_should_raise(storage, config, start_storage):
    da = DocumentArray(DocumentArray.empty(10), storage=storage, config=config)
    with pytest.raises(RuntimeError):
        da.evaluate(metrics=['precision_at_k'])


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 256}),
        ('qdrant', {'n_dim': 256}),
        ('elasticsearch', {'n_dim': 256}),
        ('redis', {'n_dim': 256}),
        ('milvus', {'n_dim': 256}),
    ],
)
def test_useless_groundtruth_warning_should_raise(storage, config, start_storage):
    da1 = DocumentArray.empty(10)
    for d in da1:
        d.tags = {'label': 'A'}
    da1.embeddings = np.random.random([10, 256])
    da1_index = DocumentArray(da1, storage=storage, config=config)
    da1.match(da1_index, exclude_self=True)
    da2 = DocumentArray.empty(10)
    with pytest.warns(UserWarning):
        da1.evaluate(ground_truth=da2, metrics=['precision_at_k'])


def dummy_embed_function(da):
    for i in range(len(da)):
        np.random.seed(int(da[i].text))
        da[i, 'embedding'] = np.random.random(5)


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 5}),
        ('qdrant', {'n_dim': 5}),
        ('elasticsearch', {'n_dim': 5}),
        ('redis', {'n_dim': 5}),
    ],
)
def test_embed_and_evaluate_single_da(storage, config, start_storage):

    gt = DocumentArray([Document(text=str(i)) for i in range(10)])
    queries_da = DocumentArray(gt, copy=True)
    queries_da = DocumentArray(queries_da, storage=storage, config=config)
    dummy_embed_function(gt)
    gt.match(gt, limit=3)

    res = queries_da.embed_and_evaluate(
        ground_truth=gt,
        metrics=['precision_at_k', 'reciprocal_rank'],
        embed_funcs=dummy_embed_function,
        match_batch_size=1,
        limit=3,
    )
    assert all([v == 1.0 for v in res.values()])


@pytest.mark.parametrize(
    'sample_size',
    [None, 10],
)
@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 5}),
        ('qdrant', {'n_dim': 5}),
        ('elasticsearch', {'n_dim': 5}),
        ('redis', {'n_dim': 5}),
    ],
)
def test_embed_and_evaluate_two_das(storage, config, sample_size, start_storage):

    gt_queries = DocumentArray([Document(text=str(i)) for i in range(100)])
    gt_index = DocumentArray([Document(text=str(i)) for i in range(100, 200)])
    queries_da = DocumentArray(gt_queries, copy=True)
    index_da = DocumentArray(gt_index, copy=True)
    index_da = DocumentArray(index_da, storage=storage, config=config)
    dummy_embed_function(gt_queries)
    dummy_embed_function(gt_index)
    gt_queries.match(gt_index, limit=3)

    res = queries_da.embed_and_evaluate(
        ground_truth=gt_queries,
        index_data=index_da,
        metrics=['precision_at_k', 'reciprocal_rank'],
        embed_funcs=dummy_embed_function,
        match_batch_size=1,
        limit=3,
        query_sample_size=sample_size,
    )
    assert all([v == 1.0 for v in res.values()])


@pytest.mark.parametrize(
    'use_index, expected, label_tag',
    [
        (False, {'precision_at_k': 1.0 / 3, 'reciprocal_rank': 1.0}, 'label'),
        (
            True,
            {'precision_at_k': 1.0 / 3, 'reciprocal_rank': 11.0 / 18.0},
            'custom_tag',
        ),
    ],
)
@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 5}),
        ('qdrant', {'n_dim': 5}),
        ('elasticsearch', {'n_dim': 5}),
        ('redis', {'n_dim': 5}),
    ],
)
def test_embed_and_evaluate_labeled_dataset(
    storage, config, start_storage, use_index, expected, label_tag
):
    metric_fns = list(expected.keys())

    def emb_func(da):
        np.random.seed(0)  # makes sure that embeddings are always equal
        da[:, 'embedding'] = np.random.random((len(da), 5))

    da1 = DocumentArray([Document(text=str(i), tags={label_tag: i}) for i in range(3)])
    da2 = DocumentArray(da1, storage=storage, config=config, copy=True)

    if (
        use_index
    ):  # query and index da are distinct # (different embeddings are generated)
        res = da1.embed_and_evaluate(
            index_data=da2,
            metrics=metric_fns,
            embed_funcs=emb_func,
            match_batch_size=1,
            limit=3,
            label_tag=label_tag,
        )
    else:  # query and index are the same (embeddings of both das are equal)
        res = da2.embed_and_evaluate(
            metrics=metric_fns,
            embed_funcs=emb_func,
            match_batch_size=1,
            limit=3,
            label_tag=label_tag,
        )
    for key in metric_fns:
        assert key in res
        assert abs(res[key] - expected[key]) < 1e-4


@pytest.mark.parametrize(
    'two_embed_funcs, kwargs',
    [
        (False, {}),
        (True, {'match_batch_size': 100}),
        (False, {'match_batch_size': 100}),
    ],
)
def test_embed_and_evaluate_on_real_data(two_embed_funcs, kwargs):
    metric_names = ['precision_at_k', 'reciprocal_rank']

    labels = ['18828_alt.atheism', '18828_comp.graphics']
    news = [load_dataset('newsgroup', label) for label in labels]
    features = [
        (data['train'][j]['text'], i)
        for i, data in enumerate(news)
        for j in range(len(data['train']))
    ]
    char_ids = {c: i for i, c in enumerate(printable)}
    np.random.shuffle(features)
    X, y = zip(*features)
    queries_x, queries_y = X[:100], y[:100]
    index_x, index_y = X[100:], y[100:]
    query_docs = DocumentArray(
        [Document(text=t, tags={'label': l}) for t, l in zip(queries_x, queries_y)]
    )
    index_docs = DocumentArray(
        [Document(text=t, tags={'label': l}) for t, l in zip(index_x, index_y)]
    )

    def emb_func(da):
        da[:, 'embedding'] = np.array(
            [[Counter(d.text)[c] for c in char_ids] for d in da], dtype='float32'
        )

    res = query_docs.embed_and_evaluate(
        index_data=index_docs,
        embed_funcs=(emb_func, emb_func) if two_embed_funcs else emb_func,
        metrics=metric_names,
        **kwargs,
    )

    # re-calculate manually
    emb_func(query_docs)
    emb_func(index_docs)
    query_docs.match(index_docs)
    res2 = query_docs.evaluate(metrics=metric_names)

    for key in res:
        assert key in res2
        assert abs(res[key] - res2[key]) < 1e-3


@pytest.fixture(scope='session')
def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 768}),
        ('qdrant', {'n_dim': 768}),
        ('elasticsearch', {'n_dim': 768}),
        ('redis', {'n_dim': 768}),
    ],
)
def test_embed_and_evaluate_with_embed_model(
    storage, config, bert_tokenizer, start_storage
):
    model = BertModel(BertConfig())
    collate_fn = lambda da: bert_tokenizer(da.texts, return_tensors='pt')
    da = DocumentArray(
        [Document(text=f'some text {i}', tags={'label': str(i)}) for i in range(5)]
    )
    da = DocumentArray(da, storage=storage, config=config)
    res = da.embed_and_evaluate(
        metrics=['precision_at_k'], embed_models=model, collate_fns=collate_fn
    )
    assert res
    assert res['precision_at_k'] == 0.2


@pytest.mark.parametrize(
    'queries, kwargs, exception',
    [
        (DocumentArray.empty(4), {}, ValueError),
        (
            DocumentArray([Document(tags={'label': 0})]),
            {'index_data': DocumentArray.empty(4)},
            ValueError,
        ),
        (DocumentArray([Document(tags={'label': 0})]), {}, RuntimeError),
        (
            DocumentArray([Document(tags={'label': 0})]),
            {'index_data': DocumentArray([Document(tags={'label': 0})])},
            RuntimeError,
        ),
    ],
)
@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 5}),
        ('qdrant', {'n_dim': 5}),
        ('elasticsearch', {'n_dim': 5}),
        ('redis', {'n_dim': 5}),
    ],
)
def test_embed_and_evaluate_invalid_input_should_raise(
    storage, config, queries, kwargs, exception, start_storage
):
    kwargs.update({'metrics': ['precision_at_k']})
    if 'index_data' in kwargs:
        kwargs['index_data'] = DocumentArray(
            kwargs['index_data'], storage=storage, config=config
        )

    with pytest.raises(exception):
        queries.embed_and_evaluate(**kwargs)


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', {}),
        ('weaviate', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 5}),
        ('qdrant', {'n_dim': 5}),
        ('elasticsearch', {'n_dim': 5}),
        ('redis', {'n_dim': 5}),
    ],
)
@pytest.mark.parametrize('sample_size', [100, 1_000, 10_000])
def test_embed_and_evaluate_sampling(storage, config, sample_size, start_storage):
    metric_fns = ['precision_at_k', 'reciprocal_rank']

    def emb_func(da):
        np.random.seed(0)  # makes sure that embeddings are always equal
        da[:, 'embedding'] = np.random.random((len(da), 5))

    da1 = DocumentArray(
        [Document(text=str(i), tags={'label': i % 20}) for i in range(2_000)]
    )
    da2 = DocumentArray(da1, storage=storage, config=config, copy=True)

    res = da1.embed_and_evaluate(
        index_data=da2,
        metrics=metric_fns,
        embed_funcs=emb_func,
        query_sample_size=sample_size,
    )
    expected_size = (
        sample_size if sample_size and (sample_size < len(da1)) else len(da1)
    )
    eval_res = [d.evaluations for d in da1 if len(d.evaluations.keys()) > 0]

    assert len(eval_res) == expected_size
    for key in res:
        assert abs(res[key] - np.mean([x[key].value for x in eval_res])) < 1e-5
