"""Which fix gives each user complete, correct results?

Every result is compared with an exact scan under the same RLS policy. Each fix runs for all
user sizes, so a fix that helps small users can't hide a slowdown for large ones.

    python src/evaluate_fixes.py --backend local
"""
import argparse
import statistics
import time
from collections import Counter

import numpy as np

import db
import fiqa
from metrics import recall_against_exact
from ownership import SEED, SHARES, assign_owners
from searches import USER_FILTER, vector

TABLE = "public.docs"
RESULTS = db.ROOT / "results"
K = 10
ROUTE_MAX_ROWS = 6000

EXACT = ["SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"]
FILTERED_EXACT = ["SET LOCAL enable_indexscan = off"]
RELAXED = ["SET LOCAL hnsw.iterative_scan = relaxed_order"]
RELAXED_MEMORY = RELAXED + ["SET LOCAL hnsw.max_scan_tuples = 100000", "SET LOCAL hnsw.scan_mem_multiplier = 4"]
SETTINGS = {"default": [], "iterative": RELAXED, "iterative + memory": RELAXED_MEMORY}

# Configurations without the owner index come first, so it is only built once.
CONFIGS = [
    dict(name="wrapped RLS", policy=db.WRAPPED_POLICY, user_filter="", owner_index=False),
    dict(name="wrapped RLS + query filter", policy=db.WRAPPED_POLICY, user_filter=USER_FILTER, owner_index=False),
    dict(name="wrapped RLS + owner index", policy=db.WRAPPED_POLICY, user_filter="", owner_index=True),
    dict(name="plain RLS + owner index", policy=db.PLAIN_POLICY, user_filter="", owner_index=True),
    dict(name="wrapped RLS + owner index + query filter", policy=db.WRAPPED_POLICY, user_filter=USER_FILTER, owner_index=True),
]


def summarise(found, exact, owned_ids):
    returned = [len(r["ids"]) for r in found.values()]
    times = [r["ms"] for r in found.values()]
    return dict(
        mean_returned=statistics.mean(returned),
        zero_results=sum(n == 0 for n in returned) / len(returned),
        short_results=sum(n < K for n in returned) / len(returned),
        recall_at_10=statistics.mean(recall_against_exact(found[q]["ids"], exact[q]["ids"]) for q in found),
        p50_ms=statistics.median(times),
        p95_ms=float(np.percentile(times, 95)),
        plans=dict(Counter(r["plan"] for r in found.values())),
        leaked_rows=sum(len(set(r["ids"]) - owned_ids) for r in found.values()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    args = parser.parse_args()
    backend = args.backend
    out = RESULTS / f"fixes_{backend}.json"
    logfile = open(RESULTS / f"fixes_{backend}.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        logfile.write(message + "\n")
        logfile.flush()

    vectors, _, _ = fiqa.documents()
    questions = fiqa.questions()
    test_qids = [q[0] for q in questions if q[1] == "test"]
    owners = assign_owners(len(vectors))
    owned_ids = {user: {i for i, o in enumerate(owners) if o == user} for user in SHARES}

    db.setup(backend)
    db.load_questions(backend, [q[:4] for q in questions])
    db.load_docs(backend, TABLE, owners, vectors, log=log)

    db.set_owner_index(backend, TABLE, False)
    db.set_policy(backend, TABLE, db.WRAPPED_POLICY)
    exact = {user: db.search(backend, user, vector(TABLE), EXACT, K, test_qids) for user in SHARES}
    log("exact scan done")

    rows = []

    def record(config, setting, user, found):
        row = dict(backend=backend, config=config, setting=setting, user=user, owned=len(owned_ids[user]),
                   **summarise(found, exact[user], owned_ids[user]))
        rows.append(row)
        log(f"{config:42} {setting:18} {user:9} returned={row['mean_returned']:5.2f} zero={row['zero_results']:6.1%} "
            f"recall={row['recall_at_10']:.3f} p50={row['p50_ms']:7.1f}ms p95={row['p95_ms']:7.1f}ms "
            f"leaked={row['leaked_rows']} plans={row['plans']}")
        db.write_json(out, dict(dataset="fiqa", documents=len(vectors), questions=len(test_qids), k=K, seed=SEED,
                                model="sentence-transformers/all-MiniLM-L6-v2", route_max_rows=ROUTE_MAX_ROWS, rows=rows))

    for config in CONFIGS:
        db.set_owner_index(backend, TABLE, config["owner_index"])
        db.set_policy(backend, TABLE, config["policy"])
        sql = vector(TABLE, config["user_filter"])
        for setting, statements in SETTINGS.items():
            for user in SHARES:
                record(config["name"], setting, user, db.search(backend, user, sql, statements, K, test_qids))

    # Routing by how many rows the user owns: filter first for small users, index for large ones.
    # The final configuration above leaves the wrapped policy and owner index in place.
    for user in SHARES:
        started = time.perf_counter()
        with db.connect(backend) as conn:
            owned = conn.execute(f"SELECT count(*) FROM {TABLE} WHERE owner = %s", (db.user_id(user),)).fetchone()[0]
        log(f"counted {owned} rows for {user} in {(time.perf_counter() - started) * 1000:.0f}ms including connection")
        statements = FILTERED_EXACT if owned <= ROUTE_MAX_ROWS else RELAXED
        record("routed by rows owned", "routed", user,
               db.search(backend, user, vector(TABLE, USER_FILTER), statements, K, test_qids))

    if any(r["leaked_rows"] for r in rows):
        raise SystemExit("A user received rows they don't own")
    log("done")


if __name__ == "__main__":
    main()
