import math

import pytest

from metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, recall_against_exact, recall_at_k, rrf_scores


def test_recall_counts_only_the_top_k():
    assert recall_at_k([1, 2, 3], [1, 3, 9, 8]) == 0.5
    assert recall_at_k([1, 2, 3], [3], k=2) == 0.0


def test_ndcg_by_hand():
    assert ndcg_at_k([1, 2], [1, 2]) == 1.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([9, 1], [1, 2]) == pytest.approx(expected)
    assert ndcg_at_k([9, 8], [1]) == 0.0


def test_mrr_by_hand():
    assert mrr_at_k([9, 1, 2], [1, 2]) == 0.5
    assert mrr_at_k([9], [1]) == 0.0
    assert mrr_at_k([9, 1], [1], k=1) == 0.0


def test_empty_results_score_zero_instead_of_passing():
    assert recall_at_k([], [1]) == 0.0
    assert ndcg_at_k([], [1]) == 0.0
    assert mrr_at_k([], [1]) == 0.0
    assert recall_against_exact([], [1, 2]) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k, mrr_at_k])
def test_a_question_without_labels_is_rejected(metric):
    with pytest.raises(ValueError):
        metric([1], [])


def test_rrf_by_hand():
    scores = rrf_scores([[1, 2], [2, 3]], k=60)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == 2


def test_bootstrap_interval_contains_the_mean_and_is_repeatable():
    values = [0.0, 1.0] * 50
    mean, low, high = bootstrap_ci(values)
    assert low < mean < high
    assert bootstrap_ci(values) == (mean, low, high)
    assert bootstrap_ci([0.5] * 10)[1:] == (0.5, 0.5)
