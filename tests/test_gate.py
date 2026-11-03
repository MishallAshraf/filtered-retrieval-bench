"""Deliberately broken results must fail the gate."""
import copy

import pytest

import gate


def relevance():
    summary = dict(n=100, ndcg_at_10=0.4, leaked_rows=0)
    return dict(dataset="fiqa", documents=10, k=10, model="m", seeds=[42], route_max_rows=6000,
                report={"hybrid": dict(all=summary, by_user={"u_1pct": dict(summary)})})


def fixes():
    return dict(dataset="fiqa", documents=10, k=10, model="m", seed=42, route_max_rows=6000,
                rows=[dict(config="c", setting="s", user="u_1pct", recall_at_10=0.98, leaked_rows=0)])


@pytest.fixture
def baseline():
    return gate.build(relevance(), fixes())


def test_unchanged_results_pass(baseline):
    assert gate.check(baseline, relevance(), fixes()) == []


def test_small_movement_within_tolerance_passes(baseline):
    results = fixes()
    results["rows"][0]["recall_at_10"] = 0.975
    assert gate.check(baseline, relevance(), results) == []


def test_lower_recall_fails(baseline):
    results = fixes()
    results["rows"][0]["recall_at_10"] = 0.5
    assert "below the baseline" in gate.check(baseline, relevance(), results)[0]


def test_lower_score_for_one_user_fails_even_if_the_average_holds(baseline):
    results = relevance()
    results["report"]["hybrid"]["by_user"]["u_1pct"]["ndcg_at_10"] = 0.1
    assert len(gate.check(baseline, results, fixes())) == 1


def test_a_leaked_row_fails(baseline):
    results = fixes()
    results["rows"][0]["leaked_rows"] = 1
    assert "don't own" in gate.check(baseline, relevance(), results)[0]


def test_a_different_setup_fails_rather_than_comparing(baseline):
    results = copy.deepcopy(relevance())
    results["model"] = "another model"
    assert "different setup" in gate.check(baseline, results, fixes())[0]


def test_missing_measurement_fails(baseline):
    results = fixes()
    results["rows"][0]["user"] = "someone else"
    assert "missing" in gate.check(baseline, relevance(), results)[0]


def test_an_empty_run_is_not_a_pass(baseline):
    results = fixes()
    results["rows"] = []
    with pytest.raises(ValueError):
        gate.check(baseline, relevance(), results)
