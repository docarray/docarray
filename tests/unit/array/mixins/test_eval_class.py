import copy

import numpy as np
import pytest

from docarray import DocumentArray, Document


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
def test_eval_mixin_perfect_match(metric_fn, kwargs):
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([10, 256])
    da.match(da, exclude_self=True)
    r = da.evaluate(da, metric=metric_fn, **kwargs)
    assert isinstance(r, float)
    assert r == 1.0
    for d in da:
        assert d.evaluations[metric_fn].value == 1.0


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
def test_eval_mixin_zero_match(metric_fn, kwargs):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 256])
    da1.match(da1, exclude_self=True)

    da2 = copy.deepcopy(da1)
    da2.embeddings = np.random.random([10, 256])
    da2.match(da2, exclude_self=True)

    r = da1.evaluate(da2, metric=metric_fn, **kwargs)
    assert isinstance(r, float)
    assert r == 1.0
    for d in da1:
        d: Document
        assert d.evaluations[metric_fn].value == 1.0


def test_diff_len_should_raise():
    da1 = DocumentArray.empty(10)
    da2 = DocumentArray.empty(5)
    with pytest.raises(ValueError):
        da1.evaluate(da2, metric='precision_at_k')


def test_diff_hash_fun_should_raise():
    da1 = DocumentArray.empty(10)
    da2 = DocumentArray.empty(10)
    with pytest.raises(ValueError):
        da1.evaluate(da2, metric='precision_at_k')


def test_same_hash_same_len_fun_should_work():
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 3])
    da1.match(da1)
    da2 = DocumentArray.empty(10)
    da2.embeddings = np.random.random([10, 3])
    da2.match(da2)
    with pytest.raises(ValueError):
        da1.evaluate(da2, metric='precision_at_k')
    for d1, d2 in zip(da1, da2):
        d1.id = d2.id

    da1.evaluate(da2, metric='precision_at_k')


def test_adding_noise():
    da = DocumentArray.empty(10)

    da.embeddings = np.random.random([10, 3])
    da.match(da, exclude_self=True)

    da2 = copy.deepcopy(da)

    for d in da2:
        d.matches.extend(DocumentArray.empty(10))
        d.matches = d.matches.shuffle()

    assert da2.evaluate(da, metric='precision_at_k', k=10) < 1.0

    for d in da2:
        assert 0.0 < d.evaluations['precision_at_k'].value < 1.0


@pytest.mark.parametrize(
    'metric_fn, kwargs',
    [
        ('recall_at_k', {}),
        ('f1_score_at_k', {}),
    ],
)
def test_diff_match_len_in_gd(metric_fn, kwargs):
    da1 = DocumentArray.empty(10)
    da1.embeddings = np.random.random([10, 128])
    da1.match(da1, exclude_self=True)

    da2 = copy.deepcopy(da1)
    da2.embeddings = np.random.random([10, 128])
    da2.match(da2, exclude_self=True)
    # pop some matches from first document
    da2[0].matches.pop(8)

    r = da1.evaluate(da2, metric=metric_fn, **kwargs)
    assert isinstance(r, float)
    np.testing.assert_allclose(r, 1.0, rtol=1e-2)  #
    for d in da1:
        d: Document
        # f1_score does not yield 1 for the first document as one of the match is missing
        assert d.evaluations[metric_fn].value > 0.9
