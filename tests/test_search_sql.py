"""Run the real search SQL and RLS on a small database where every rank can be worked out by hand.

Four users own the rows below. Vectors point along one axis, so distance order is obvious.
"""
import numpy as np
import pytest

import db
import searches
from metrics import rrf_scores

TABLE = "public.tiny_docs"
QIDS = [9001, 9002]


def unit(x):
    v = np.zeros(384, dtype=np.float32)
    v[0], v[1] = x, (1 - x * x) ** 0.5
    return v


# id: owner, closeness to the question, text.
DOCS = [
    ("alice", 0.99, "gardening notes about tomatoes"),
    ("bob", 0.98, "dividend tax rules"),
    ("alice", 0.90, "dividend tax on shares"),
    ("alice", 0.50, "unrelated recipe"),
    ("bob", 0.40, "dividend tax treaty"),
]


@pytest.fixture(scope="module")
def backend():
    db.setup("local")
    with db.connect("local") as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute("DELETE FROM bench.questions WHERE qid >= 9000")
    db.load_docs("local", TABLE, [d[0] for d in DOCS], [unit(d[1]) for d in DOCS],
                 [d[2] for d in DOCS], log=lambda _: None)
    db.set_policy("local", TABLE, db.WRAPPED_POLICY)
    with db.connect("local") as conn:
        for qid, question in zip(QIDS, ["dividend tax", "dividend chicken"]):
            conn.execute("INSERT INTO bench.questions VALUES (%s, 'tiny', %s, %s)", (qid, question, unit(1.0)))
    yield "local"
    with db.connect("local") as conn:
        conn.execute(f"DROP TABLE {TABLE}")
        conn.execute("DELETE FROM bench.questions WHERE qid >= 9000")


EXACT = ["SET LOCAL enable_indexscan = off"]


def run(backend, user, sql, qid=QIDS[0], k=10):
    return db.search(backend, user, sql, EXACT, k=k, qids=[qid])[qid]["ids"]


def test_rls_hides_other_users_rows(backend):
    assert run(backend, "alice", searches.vector(TABLE)) == [0, 2, 3]
    assert run(backend, "bob", searches.vector(TABLE)) == [1, 4]
    assert run(backend, "nobody", searches.vector(TABLE)) == []


def test_keyword_and_needs_every_term_while_or_needs_one(backend):
    assert run(backend, "alice", searches.keyword(TABLE, "and"), QIDS[1]) == []
    assert run(backend, "alice", searches.keyword(TABLE, "or"), QIDS[1]) == [2]


def test_minimum_keyword_score_removes_a_weak_match(backend):
    assert run(backend, "alice", searches.keyword(TABLE, "and")) == [2]
    assert run(backend, "alice", searches.keyword(TABLE, "and", min_score=100)) == []


def test_hybrid_sql_matches_hand_computed_rrf(backend):
    vector_ranking = [0, 2, 3]
    keyword_ranking = [2]
    scores = rrf_scores([vector_ranking, keyword_ranking])
    expected = sorted(scores, key=lambda d: (-scores[d], d))
    assert expected == [2, 0, 3]
    assert run(backend, "alice", searches.hybrid(TABLE, "and")) == expected


def test_hybrid_stays_within_the_user_rows(backend):
    assert set(run(backend, "bob", searches.hybrid(TABLE, "or"))) <= {1, 4}


def test_plan_is_reported_for_the_query_that_ran(backend):
    forced = db.search(backend, "alice", searches.vector(TABLE), EXACT, qids=[QIDS[0]])
    assert "hnsw" not in forced[QIDS[0]]["plan"]
    allowed = db.search(backend, "alice", searches.vector(TABLE),
                        ["SET LOCAL enable_seqscan = off"], qids=[QIDS[0]])
    assert "hnsw" in allowed[QIDS[0]]["plan"]
