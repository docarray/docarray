import copy

import numpy as np
import pytest

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
    r = da1.evaluate(metric_fn, **kwargs)[metric_fn]
    assert isinstance(r, float)
    assert r == 0.0
    for d in da1:
        assert d.evaluations[metric_fn].value == 0.0


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
def test_eval_mixin_one_of_n_labeled(metric_fn, metric_score):
    da = DocumentArray([Document(text=str(i), tags={'label': i}) for i in range(3)])
    for d in da:
        d.matches = da
    r = da.evaluate(metric_fn)[metric_fn]
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
