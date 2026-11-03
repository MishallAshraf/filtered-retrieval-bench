"""Are the results useful? Score search methods against FiQA's human relevance labels.

Settings are chosen on dev questions with the first seed, then fixed and reported on test
questions for every seed. Each user is only asked questions they own relevant documents for.
Placing those documents with the user can flatter these scores, so they compare methods and
don't predict production quality.

    python src/evaluate_relevance.py --backend local --seeds 42 43 44
"""
import argparse
import re
import statistics
from collections import Counter

import numpy as np

import db
import fiqa
from metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, recall_at_k
from ownership import LABELLED_USERS, assign_labelled
from searches import USER_FILTER, hybrid, keyword, vector

TABLE = "public.docs_labelled"
RESULTS = db.ROOT / "results"
K = 10
ROUTE_MAX_ROWS = 6000
RARE_MAX_DOCS = 5
MIN_SCORES = (0.0, 0.01, 0.1)

FILTERED_EXACT = ["SET LOCAL enable_indexscan = off"]
RELAXED = ["SET LOCAL hnsw.iterative_scan = relaxed_order"]


def routed(owned):
    return FILTERED_EXACT if owned <= ROUTE_MAX_ROWS else RELAXED


def rare_questions(rows, texts):
    """Question ids containing a word found in very few documents, such as a product name."""
    tokens = re.compile(r"[a-z0-9]+")
    counts = Counter()
    for text in texts:
        counts.update(set(tokens.findall(text.lower())))
    return {row[4] for row in rows
            if any(0 < counts[t] <= RARE_MAX_DOCS for t in set(tokens.findall(row[2].lower())))}


def tuning_grid():
    grid = {}
    for operator in ("and", "or"):
        for score in MIN_SCORES:
            grid[f"keyword {operator} min={score}"] = ("keyword", operator, score)
            grid[f"hybrid {operator} min={score}"] = ("hybrid", operator, score)
    return grid


def build_sql(kind, operator, score):
    build = keyword if kind == "keyword" else hybrid
    return build(TABLE, operator, USER_FILTER, score)


def score_run(found, labelled, qid_of, owned_ids):
    """Per-question scores for one search method and one user."""
    scored = {}
    for question, relevant in labelled.items():
        result = found[qid_of[question]]
        scored[question] = dict(
            ndcg=ndcg_at_k(result["ids"], relevant, K), recall=recall_at_k(result["ids"], relevant, K),
            mrr=mrr_at_k(result["ids"], relevant, K), returned=len(result["ids"]), ms=result["ms"],
            plan=result["plan"], leaked=len(set(result["ids"]) - owned_ids))
    return scored


def run_method(backend, user, sql, settings, labelled, qid_of, owned_ids):
    qids = [qid_of[q] for q in labelled]
    return score_run(db.search(backend, user, sql, settings, K, qids), labelled, qid_of, owned_ids)


def summarise(scores):
    values = list(scores)
    ndcg, low, high = bootstrap_ci([s["ndcg"] for s in values])
    return dict(n=len(values), ndcg_at_10=ndcg, ndcg_ci=[low, high],
                recall_at_10=statistics.mean(s["recall"] for s in values),
                mrr_at_10=statistics.mean(s["mrr"] for s in values),
                mean_returned=statistics.mean(s["returned"] for s in values),
                no_results=sum(s["returned"] == 0 for s in values) / len(values),
                p50_ms=statistics.median(s["ms"] for s in values),
                p95_ms=float(np.percentile([s["ms"] for s in values], 95)),
                plans=dict(Counter(s["plan"] for s in values)),
                leaked_rows=sum(s["leaked"] for s in values))


def paired_difference(a, b):
    """nDCG of a minus b, with a confidence interval over the same questions."""
    mean, low, high = bootstrap_ci([x["ndcg"] - y["ndcg"] for x, y in zip(a, b)])
    return dict(mean=mean, ci=[low, high], n=len(a))


def reload_table(backend, owners, vectors, texts, log):
    with db.connect(backend) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    db.load_docs(backend, TABLE, owners, vectors, texts, log=log)
    db.set_policy(backend, TABLE, db.WRAPPED_POLICY)
    db.set_owner_index(backend, TABLE, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    backend = args.backend
    logfile = open(RESULTS / f"relevance_{backend}.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        logfile.write(message + "\n")
        logfile.flush()

    vectors, texts, doc_index = fiqa.documents()
    rows = fiqa.questions()
    qid_of = {row[4]: row[0] for row in rows}
    rare = rare_questions(rows, texts)
    labels = {split: fiqa.labels(split, doc_index) for split in fiqa.SPLITS}

    db.setup(backend)
    db.load_questions(backend, [row[:4] for row in rows])

    chosen = None
    all_scores = {}
    per_seed = {}
    for seed in args.seeds:
        owners, assignments = assign_labelled(len(vectors), labels, seed=seed)
        owned_ids = {user: {i for i, o in enumerate(owners) if o == user} for user in LABELLED_USERS}
        kept = {user: {split: len(assignments[user][split]) for split in fiqa.SPLITS} for user in LABELLED_USERS}
        log(f"seed {seed}: questions kept {kept}")
        reload_table(backend, owners, vectors, texts, log)

        if chosen is None:
            dev_scores = {}
            for name, (kind, operator, score) in tuning_grid().items():
                combined = []
                for user in LABELLED_USERS:
                    combined += run_method(backend, user, build_sql(kind, operator, score), routed(len(owned_ids[user])),
                                           assignments[user]["dev"], qid_of, owned_ids[user]).values()
                dev_scores[name] = summarise(combined)
                log(f"dev {name:28} nDCG@10={dev_scores[name]['ndcg_at_10']:.3f} "
                    f"no results={dev_scores[name]['no_results']:.1%}")
            best = lambda kind: max((n for n in dev_scores if n.startswith(kind)), key=lambda n: dev_scores[n]["ndcg_at_10"])
            chosen = dict(keyword=best("keyword"), hybrid=best("hybrid"), keyword_and="keyword and min=0.0",
                          hybrid_and="hybrid and min=0.0", hybrid_or="hybrid or min=0.0", dev=dev_scores)
            log(f"chosen on dev: keyword={chosen['keyword']} hybrid={chosen['hybrid']}")

        grid = tuning_grid()
        methods = {
            "vector, no fix": (vector(TABLE), lambda owned: []),
            "vector, exact": (vector(TABLE, USER_FILTER), lambda owned: FILTERED_EXACT),
            "vector, fixed": (vector(TABLE, USER_FILTER), routed),
            "keyword": (build_sql(*grid[chosen["keyword"]]), routed),
            "hybrid": (build_sql(*grid[chosen["hybrid"]]), routed),
        }
        per_seed[seed] = {}
        for method, (sql, settings) in methods.items():
            for user in LABELLED_USERS:
                scored = run_method(backend, user, sql, settings(len(owned_ids[user])),
                                    assignments[user]["test"], qid_of, owned_ids[user])
                all_scores.setdefault(method, {}).setdefault(user, []).extend(
                    dict(s, question=q, seed=seed, rare=q in rare) for q, s in scored.items())
                summary = summarise(scored.values())
                per_seed[seed].setdefault(method, {})[user] = summary
                log(f"seed {seed} {method:16} {user:9} n={summary['n']:3} nDCG@10={summary['ndcg_at_10']:.3f} "
                    f"recall={summary['recall_at_10']:.3f} returned={summary['mean_returned']:5.2f} "
                    f"no results={summary['no_results']:6.1%} p50={summary['p50_ms']:7.1f}ms leaked={summary['leaked_rows']}")

    report = {}
    for method, by_user in all_scores.items():
        everyone = [s for scores in by_user.values() for s in scores]
        report[method] = dict(
            all=summarise(everyone),
            by_user={user: summarise(scores) for user, scores in by_user.items()},
            rare_terms=summarise([s for s in everyone if s["rare"]]),
            common_terms=summarise([s for s in everyone if not s["rare"]]),
        )

    def flat(method, keep=lambda s: True):
        return [s for scores in all_scores[method].values() for s in scores if keep(s)]

    comparisons = {
        "hybrid minus vector fixed, all": paired_difference(flat("hybrid"), flat("vector, fixed")),
        "hybrid minus vector fixed, rare terms": paired_difference(flat("hybrid", lambda s: s["rare"]), flat("vector, fixed", lambda s: s["rare"])),
        "vector fixed minus vector no fix, all": paired_difference(flat("vector, fixed"), flat("vector, no fix")),
    }
    db.write_json(RESULTS / f"relevance_{backend}.json", dict(
        dataset="fiqa", documents=len(vectors), k=K, seeds=args.seeds, route_max_rows=ROUTE_MAX_ROWS,
        rare_max_docs=RARE_MAX_DOCS, model="sentence-transformers/all-MiniLM-L6-v2",
        chosen_on_dev={k: v for k, v in chosen.items() if k != "dev"}, dev=chosen["dev"],
        report=report, comparisons=comparisons, per_seed=per_seed))
    for name, value in comparisons.items():
        log(f"{name}: {value['mean']:+.3f} [{value['ci'][0]:+.3f}, {value['ci'][1]:+.3f}] n={value['n']}")
    if any(s["leaked"] for m in all_scores for scores in all_scores[m].values() for s in scores):
        raise SystemExit("A user received rows they don't own")
    log("done")


if __name__ == "__main__":
    main()
