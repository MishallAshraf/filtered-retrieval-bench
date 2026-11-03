"""Check the recommended search function on the guide's schema, written the way an app would call it.

Each form runs as the authenticated role through the guide's policy and is compared with an exact
scan. It needs the tables from evaluate_guide_schema.py. Functions are dropped afterwards, so they
aren't left exposed as API endpoints on a hosted project.

    python scripts/verify_guide_recipe.py --backend supabase
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db  # noqa: E402
import fiqa  # noqa: E402
from evaluate_guide_schema import (DOCUMENTS, OWNERS, SECTIONS, VARIANTS, group_documents, out_of_order,  # noqa: E402
                                   search_sql, set_indexes)
from metrics import recall_against_exact  # noqa: E402
from ownership import SHARES  # noqa: E402

QIDS = list(range(100))
EXACT = ["SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"]
SIGNATURE = "(query_embedding vector(384), match_count int) RETURNS TABLE (id int, distance double precision)"
ITERATIVE = "SET hnsw.iterative_scan = relaxed_order"
LOOKUP = {
    "single owner": f"SELECT d.id FROM {DOCUMENTS} d WHERE d.owner_id = auth.uid()",
    "multiple owners": f"SELECT o.document_id FROM {OWNERS} o WHERE o.owner_id = auth.uid()",
}


def body(lookup, vector, limit):
    return (f"WITH nearest AS MATERIALIZED (SELECT s.id, s.embedding <=> {vector} AS distance FROM {SECTIONS} s "
            f"WHERE s.document_id IN ({lookup}) ORDER BY distance LIMIT {limit}) "
            "SELECT nearest.id, nearest.distance FROM nearest ORDER BY nearest.distance + 0")


def recipes(lookup):
    static = body(lookup, "query_embedding", "match_count")
    dynamic = body(lookup, "$1", "$2").replace("'", "''")
    return {
        "sql function": f"LANGUAGE sql STABLE {ITERATIVE} AS $$ {static} $$",
        "plpgsql function": f"LANGUAGE plpgsql STABLE {ITERATIVE} AS $$ BEGIN RETURN QUERY {static}; END $$",
        "plpgsql function, dynamic query": (
            f"LANGUAGE plpgsql STABLE {ITERATIVE} AS $$ BEGIN RETURN QUERY EXECUTE '{dynamic}' "
            "USING query_embedding, match_count; END $$"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    backend = parser.parse_args().backend
    vectors, _, _ = fiqa.documents()
    query_vectors = {q[0]: q[3] for q in fiqa.questions()}
    document_of, document_owners = group_documents(len(vectors))
    owned_ids = {user: {i for i, d in enumerate(document_of) if document_owners[d] == user} for user in SHARES}

    rows = []
    for variant, spec in VARIANTS.items():
        set_indexes(backend, [f"CREATE INDEX guide_sections_docid ON {SECTIONS} (document_id)", spec["lookup_index"]])
        db.set_policy(backend, SECTIONS, spec["policy"])
        exact = {user: db.search(backend, user, search_sql(), EXACT, 10, QIDS) for user in SHARES}
        for name, recipe in recipes(LOOKUP[variant]).items():
            with db.connect(backend) as conn:
                conn.execute("DROP FUNCTION IF EXISTS public.match_sections(vector, int)")
                conn.execute(f"CREATE FUNCTION public.match_sections{SIGNATURE} {recipe}")
                conn.execute("GRANT EXECUTE ON FUNCTION public.match_sections(vector, int) TO authenticated")
            try:
                for user in SHARES:
                    found = db.search(backend, user, "SELECT id FROM public.match_sections($1, $2)", (), 10, QIDS)
                    row = dict(variant=variant, recipe=name, user=user,
                               returned=statistics.mean(len(found[q]["ids"]) for q in QIDS),
                               recall_at_10=statistics.mean(recall_against_exact(found[q]["ids"], exact[user][q]["ids"]) for q in QIDS),
                               p50_ms=statistics.median(found[q]["ms"] for q in QIDS),
                               p95_ms=float(np.percentile([found[q]["ms"] for q in QIDS], 95)),
                               out_of_order=out_of_order(found, vectors, query_vectors),
                               leaked_rows=sum(len(set(found[q]["ids"]) - owned_ids[user]) for q in QIDS))
                    rows.append(row)
                    print(f"{variant:15} {name:32} {user:9} returned={row['returned']:5.2f} recall={row['recall_at_10']:.3f} "
                          f"p50={row['p50_ms']:.1f}ms p95={row['p95_ms']:.1f}ms unordered={row['out_of_order']:.1%} "
                          f"leaked={row['leaked_rows']}", flush=True)
            finally:
                with db.connect(backend) as conn:
                    conn.execute("DROP FUNCTION IF EXISTS public.match_sections(vector, int)")
    set_indexes(backend, [])
    db.write_json(db.ROOT / "results" / f"guide_recipes_{backend}.json", dict(questions=len(QIDS), k=10, rows=rows))
    return 1 if any(r["leaked_rows"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
