"""Check search functions written the way an app would call them, on either backend.

The benchmark runs each query with its values known in advance, but a function usually hands the
planner parameters instead, and that can change the plan. Each function runs as the authenticated
role through RLS and is compared with an exact scan. Functions are dropped afterwards, so they
aren't left exposed as API endpoints on a hosted project.

    python scripts/verify_recipe.py --backend supabase
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db  # noqa: E402
from metrics import recall_against_exact  # noqa: E402
from ownership import SHARES, assign_owners  # noqa: E402
from searches import vector  # noqa: E402

TABLE = "public.docs"
QIDS = list(range(100))
EXACT = ["SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"]
SIGNATURE = "(query_embedding vector(384), match_count int) RETURNS TABLE (id int)"
ITERATIVE = "SET hnsw.iterative_scan = relaxed_order"
MEMORY = "SET hnsw.max_scan_tuples = 100000 SET hnsw.scan_mem_multiplier = 4"
BODY = """SELECT d.id FROM public.docs d WHERE d.owner = auth.uid()
          ORDER BY d.embedding <=> query_embedding LIMIT match_count"""
DYNAMIC = """SELECT d.id FROM public.docs d WHERE d.owner = auth.uid()
             ORDER BY d.embedding <=> $1 LIMIT $2"""

RECIPES = {
    "sql function": f"LANGUAGE sql STABLE {ITERATIVE} AS $$ {BODY} $$",
    "sql function, more scan memory": f"LANGUAGE sql STABLE {ITERATIVE} {MEMORY} AS $$ {BODY} $$",
    "plpgsql function": f"LANGUAGE plpgsql STABLE {ITERATIVE} AS $$ BEGIN RETURN QUERY {BODY}; END $$",
    "plpgsql function, dynamic query": (
        f"LANGUAGE plpgsql STABLE {ITERATIVE} AS $$ BEGIN RETURN QUERY EXECUTE '{DYNAMIC}' "
        "USING query_embedding, match_count; END $$"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    backend = parser.parse_args().backend
    with db.connect(backend) as conn:
        documents = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    owners = assign_owners(documents)
    owned_ids = {user: {i for i, o in enumerate(owners) if o == user} for user in SHARES}

    db.set_owner_index(backend, TABLE, True)
    db.set_policy(backend, TABLE, db.WRAPPED_POLICY)
    exact = {user: db.search(backend, user, vector(TABLE), EXACT, 10, QIDS) for user in SHARES}
    rows = []
    for name, recipe in RECIPES.items():
        with db.connect(backend) as conn:
            conn.execute("DROP FUNCTION IF EXISTS public.match_docs(vector, int)")
            conn.execute(f"CREATE FUNCTION public.match_docs{SIGNATURE} {recipe}")
            conn.execute("GRANT EXECUTE ON FUNCTION public.match_docs(vector, int) TO authenticated")
        try:
            for user in SHARES:
                found = db.search(backend, user, "SELECT id FROM public.match_docs($1, $2)", (), 10, QIDS)
                row = dict(recipe=name, user=user, returned=statistics.mean(len(found[q]["ids"]) for q in QIDS),
                           recall_at_10=statistics.mean(recall_against_exact(found[q]["ids"], exact[user][q]["ids"]) for q in QIDS),
                           p50_ms=statistics.median(found[q]["ms"] for q in QIDS),
                           leaked_rows=sum(len(set(found[q]["ids"]) - owned_ids[user]) for q in QIDS))
                rows.append(row)
                print(f"{name:34} {user:9} returned={row['returned']:5.2f} recall={row['recall_at_10']:.3f} "
                      f"p50={row['p50_ms']:.1f}ms leaked={row['leaked_rows']}", flush=True)
        finally:
            with db.connect(backend) as conn:
                conn.execute("DROP FUNCTION IF EXISTS public.match_docs(vector, int)")
    db.write_json(db.ROOT / "results" / f"recipes_{backend}.json", dict(questions=len(QIDS), k=10, rows=rows))
    return 1 if any(r["leaked_rows"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
