import pytest

from docarray.math.evaluation import (
    average_precision,
    dcg_at_k,
    f1_score_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
)


@pytest.mark.parametrize(
    "binary_relevance, score",
    [
        ([0, 1, 0, 0, 1, 1, 1], 0.25),
        ([], 0),
        ([1, 1, 1], 1),
        ([0, 0], 0),
    ],
)
def test_r_precision(binary_relevance, score):
    assert abs(r_precision(binary_relevance) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 4.0 / 7, None),
        ([0, 1, 0, 0, 1, 1, 1], 0.5, 2),
        ([], 0, None),
        ([1, 1, 1], 1, None),
        ([0, 0], 0, None),
    ],
)
def test_precision_at_k(binary_relevance, score, k):
    assert abs(precision_at_k(binary_relevance, k=k) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 1, None),
        ([0, 1, 0, 0, 1, 1, 1], 0, 1),
        ([], 0, None),
        ([1, 1, 1], 1, None),
        ([0, 0], 0, None),
    ],
)
def test_hit_at_k(binary_relevance, score, k):
    assert abs(hit_at_k(binary_relevance, k=k) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score",
    [
        ([0, 1, 0, 0, 1, 1, 1], (1.0 / 2 + 2.0 / 5 + 3.0 / 6 + 4.0 / 7) / 4),
        ([], 0),
        ([1, 1, 1], 1),
        ([0, 0], 0),
    ],
)
def test_average_precision(binary_relevance, score):
    assert abs(average_precision(binary_relevance) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score",
    [
        ([0, 1, 0, 0, 1, 1, 1], 0.5),
        ([], 0),
        ([1, 1, 1], 1.0),
        ([0, 0], 0),
    ],
)
def test_reciprocal_rank(binary_relevance, score):
    assert abs(reciprocal_rank(binary_relevance) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, max_rel, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 4.0 / 7, 7, None),
        ([0, 1, 0, 0, 1, 1, 1], 1, 4, None),
        ([0, 1, 0, 0, 1, 1, 1], 0.25, 4, 2),
        ([], 0, 4, None),
        ([1, 1, 1], 0.75, 4, None),
        ([0, 0], 0, 4, None),
    ],
)
def test_recall_at_k(binary_relevance, score, max_rel, k):
    calculated_score = recall_at_k(binary_relevance, max_rel=max_rel, k=k)
    assert abs(calculated_score - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, max_rel, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 4.0 / 7, 7, None),
        ([0, 1, 0, 0, 1, 1, 1], 2 / (1 / (4 / 7) + 1), 4, None),
        ([0, 1, 0, 0, 1, 1, 1], 2 / (1 / 0.5 + 1 / 0.25), 4, 2),
        ([], 0, 4, None),
        ([1, 1, 1], 2 / (1 / 0.75 + 1), 4, None),
        ([0, 0], 0, 4, None),
    ],
)
def test_f1_score_at_k(binary_relevance, score, max_rel, k):
    calculated_score = f1_score_at_k(binary_relevance, max_rel=max_rel, k=k)
    assert abs(calculated_score - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, method, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 2.1737, 0, None),
        ([0, 1, 0, 0, 1, 1, 1], 1.7073, 1, None),
        ([0, 1, 0, 0, 1, 1, 1], 1, 0, 4),
        ([], 0, 0, None),
        ([1, 1, 1], 2.6309, 0, None),
        ([0, 0], 0, 0, None),
    ],
)
def test_dcg_at_k(binary_relevance, score, method, k):
    assert abs(dcg_at_k(binary_relevance, method=method, k=k) - score) < 0.001


@pytest.mark.parametrize(
    "binary_relevance, score, method, k",
    [
        ([0, 1, 0, 0, 1, 1, 1], 0.6942, 0, None),
        ([0, 1, 0, 0, 1, 1, 1], 0.6665, 1, None),
        ([0, 1, 0, 0, 1, 1, 1], 0.3194, 0, 4),
        ([], 0, 0, None),
        ([1, 1, 1], 1, 0, None),
        ([0, 0], 0, 0, None),
    ],
)
def test_ndcg_at_k(binary_relevance, score, method, k):
    assert abs(ndcg_at_k(binary_relevance, method=method, k=k) - score) < 0.001
