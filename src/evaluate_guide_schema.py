"""Does the shortfall, and the fix, carry over to the schema in Supabase's RAG with Permissions guide?

The guide keeps ownership on a parent documents table. Sections have no owner column, and the
policy checks each section's document_id against the user's documents. Its multiple-owner
version checks a document_owners table instead. The same FiQA texts and embeddings are used as
sections, grouped at random into documents of 8 sections, and documents are assigned to users
in the same shares as before. Every result is compared with an exact scan under the same policy.

    python src/evaluate_guide_schema.py --backend local
"""
import argparse
import random
import time

import numpy as np

import db
import fiqa
from evaluate_fixes import EXACT, FILTERED_EXACT, K, RELAXED, RELAXED_MEMORY, ROUTE_MAX_ROWS, summarise
from ownership import SEED, SHARES, assign_owners

RESULTS = db.ROOT / "results"
SECTIONS = "public.guide_sections"
DOCUMENTS = "public.guide_documents"
OWNERS = "public.guide_document_owners"
SECTIONS_PER_DOCUMENT = 8
# The guide's own query has a similarity threshold and no LIMIT.
MAX_DISTANCE = 0.5

VARIANTS = {
    "single owner": dict(
        policy=f"document_id IN (SELECT id FROM {DOCUMENTS} WHERE owner_id = (select auth.uid()))",
        user_filter=f"document_id IN (SELECT id FROM {DOCUMENTS} WHERE owner_id = auth.uid())",
        lookup_index=f"CREATE INDEX IF NOT EXISTS guide_documents_owner ON {DOCUMENTS} (owner_id)"),
    "multiple owners": dict(
        policy=f"document_id IN (SELECT document_id FROM {OWNERS} WHERE owner_id = (select auth.uid()))",
        user_filter=f"document_id IN (SELECT document_id FROM {OWNERS} WHERE owner_id = auth.uid())",
        lookup_index=f"CREATE INDEX IF NOT EXISTS guide_document_owners_owner ON {OWNERS} (owner_id, document_id)"),
}
OPTIONAL_INDEXES = ("guide_sections_docid", "guide_sections_owner", "guide_documents_owner", "guide_document_owners_owner")


def search_sql(where="", limit=True):
    clauses = [c for c in (where, f"embedding <=> $1 < {MAX_DISTANCE}" if not limit else "") if c]
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return f"SELECT id FROM {SECTIONS} {where} ORDER BY embedding <=> $1" + (" LIMIT $2" if limit else "")


def resorted_sql(where):
    """pgvector's recipe for putting relaxed_order results back in distance order."""
    return (f"SELECT id FROM (WITH nearest AS MATERIALIZED (SELECT id, embedding <=> $1 AS distance FROM {SECTIONS} "
            f"WHERE {where} ORDER BY distance LIMIT $2) SELECT id, distance FROM nearest ORDER BY distance + 0) s")


def group_documents(n):
    """Return (document of each section, owner of each document)."""
    ids = list(range(n))
    random.Random(SEED + 1).shuffle(ids)
    document_of = [0] * n
    for position, section in enumerate(ids):
        document_of[section] = position // SECTIONS_PER_DOCUMENT
    return document_of, assign_owners(-(-n // SECTIONS_PER_DOCUMENT))


def load(backend, document_of, document_owners, log):
    """Build the guide's tables, copying embeddings from public.docs so nothing is re-embedded."""
    with db.connect(backend) as conn:
        if conn.execute(f"SELECT to_regclass('{SECTIONS}') IS NOT NULL").fetchone()[0] and \
                conn.execute(f"SELECT count(*) FROM {SECTIONS}").fetchone()[0] == len(document_of):
            log("guide tables already loaded")
            return
        started = time.time()
        conn.execute(f"DROP TABLE IF EXISTS {SECTIONS}, {OWNERS}, {DOCUMENTS}")
        conn.execute(f"CREATE TABLE {DOCUMENTS} (id int PRIMARY KEY, name text NOT NULL, owner_id uuid NOT NULL)")
        conn.execute(f"CREATE TABLE {OWNERS} (id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY, owner_id uuid NOT NULL, "
                     f"document_id int NOT NULL REFERENCES {DOCUMENTS} (id))")
        # owner is the denormalised copy that one fix adds; the guide's policies never read it.
        conn.execute(f"CREATE TABLE {SECTIONS} (id int PRIMARY KEY, document_id int NOT NULL REFERENCES {DOCUMENTS} (id), "
                     "owner uuid NOT NULL, embedding vector(384) NOT NULL)")
        with conn.cursor().copy(f"COPY {DOCUMENTS} (id, name, owner_id) FROM STDIN") as copy:
            for i, owner in enumerate(document_owners):
                copy.write_row((i, f"document {i}", db.user_id(owner)))
        conn.execute(f"INSERT INTO {OWNERS} (owner_id, document_id) SELECT owner_id, id FROM {DOCUMENTS}")
        conn.execute("CREATE TEMP TABLE section_map (id int PRIMARY KEY, document_id int NOT NULL)")
        with conn.cursor().copy("COPY section_map (id, document_id) FROM STDIN") as copy:
            for i, document in enumerate(document_of):
                copy.write_row((i, document))
        conn.execute(f"INSERT INTO {SECTIONS} (id, document_id, owner, embedding) SELECT d.id, m.document_id, g.owner_id, "
                     f"d.embedding FROM public.docs d JOIN section_map m USING (id) JOIN {DOCUMENTS} g ON g.id = m.document_id")
        copied = conn.execute(f"SELECT count(*) FROM {SECTIONS}").fetchone()[0]
        if copied != len(document_of):
            raise SystemExit(f"copied {copied} sections, expected {len(document_of)}; is public.docs loaded?")
        conn.commit()
        conn.execute(f"SET maintenance_work_mem = '{'1GB' if backend == 'local' else '64MB'}'")
        if backend != "local":
            conn.execute("SET max_parallel_maintenance_workers = 0")
        conn.execute(f"CREATE INDEX guide_sections_embedding ON {SECTIONS} USING hnsw (embedding vector_cosine_ops)")
        for table in (DOCUMENTS, OWNERS, SECTIONS):
            conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            conn.execute(f"GRANT SELECT ON {table} TO authenticated")
            conn.execute(f"ANALYZE {table}")
        conn.execute(f"CREATE POLICY own_rows ON {DOCUMENTS} FOR SELECT TO authenticated USING (owner_id = (select auth.uid()))")
        conn.execute(f"CREATE POLICY own_rows ON {OWNERS} FOR SELECT TO authenticated USING (owner_id = (select auth.uid()))")
        conn.commit()
        log(f"guide tables loaded in {time.time() - started:.0f}s; database "
            + conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0])


def set_indexes(backend, statements):
    with db.connect(backend) as conn:
        for name in OPTIONAL_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS public.{name}")
        for statement in statements:
            conn.execute(statement)
        for table in (DOCUMENTS, OWNERS, SECTIONS):
            conn.execute(f"ANALYZE {table}")


def out_of_order(found, vectors, query_vectors):
    """Share of questions whose results aren't in distance order."""
    late = 0
    for qid, result in found.items():
        if len(result["ids"]) > 1:
            distances = 1 - vectors[result["ids"]] @ query_vectors[qid]
            late += bool(np.any(np.diff(distances) < -1e-6))
    return late / len(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    backend = parser.parse_args().backend
    out = RESULTS / f"guide_schema_{backend}.json"
    logfile = open(RESULTS / f"guide_schema_{backend}.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        logfile.write(message + "\n")
        logfile.flush()

    vectors, _, _ = fiqa.documents()
    questions = fiqa.questions()
    test_qids = [q[0] for q in questions if q[1] == "test"]
    query_vectors = {q[0]: q[3] for q in questions}
    document_of, document_owners = group_documents(len(vectors))
    owned_ids = {user: {i for i, d in enumerate(document_of) if document_owners[d] == user} for user in SHARES}

    db.setup(backend)
    db.load_questions(backend, [q[:4] for q in questions])
    load(backend, document_of, document_owners, log)
    rows = []

    def run(variant, config, setting, user, sql, statements, exact):
        found = db.search(backend, user, sql, statements, K, test_qids)
        returned = {q: len(r["ids"]) for q, r in found.items()}
        for r in found.values():
            r["ids"] = r["ids"][:K]
        row = dict(backend=backend, variant=variant, config=config, setting=setting, user=user,
                   owned=len(owned_ids[user]), **summarise(found, exact, owned_ids[user]),
                   out_of_order=out_of_order(found, vectors, query_vectors))
        if sql.endswith("< %s ORDER BY embedding <=> $1" % MAX_DISTANCE):
            row["mean_rows_before_top_10"] = float(np.mean(list(returned.values())))
        rows.append(row)
        log(f"{variant:15} {config:38} {setting:18} {user:9} returned={row['mean_returned']:5.2f} "
            f"zero={row['zero_results']:6.1%} recall={row['recall_at_10']:.3f} p50={row['p50_ms']:7.1f}ms "
            f"p95={row['p95_ms']:7.1f}ms unordered={row['out_of_order']:5.1%} leaked={row['leaked_rows']} plans={row['plans']}")
        db.write_json(out, dict(dataset="fiqa", sections=len(vectors), sections_per_document=SECTIONS_PER_DOCUMENT,
                                questions=len(test_qids), k=K, seed=SEED, max_distance=MAX_DISTANCE,
                                model="sentence-transformers/all-MiniLM-L6-v2", route_max_rows=ROUTE_MAX_ROWS, rows=rows))

    for variant, spec in VARIANTS.items():
        set_indexes(backend, [])
        db.set_policy(backend, SECTIONS, spec["policy"])
        exact = {user: db.search(backend, user, search_sql(), EXACT, K, test_qids) for user in SHARES}
        exact_threshold = {user: db.search(backend, user, search_sql(limit=False), EXACT, K, test_qids) for user in SHARES}
        for found in exact_threshold.values():
            for r in found.values():
                r["ids"] = r["ids"][:K]
        log(f"{variant}: exact scans done")

        for user in SHARES:
            run(variant, "guide query, no LIMIT", "default", user, search_sql(limit=False), [], exact_threshold[user])
        for setting, statements in {"default": [], "iterative": RELAXED, "iterative + memory": RELAXED_MEMORY}.items():
            for user in SHARES:
                run(variant, "guide policy only", setting, user, search_sql(), statements, exact[user])

        set_indexes(backend, [f"CREATE INDEX guide_sections_docid ON {SECTIONS} (document_id)", spec["lookup_index"]])
        for setting, statements in {"default": [], "iterative": RELAXED}.items():
            for user in SHARES:
                run(variant, "document_id filter + indexes", setting, user, search_sql(spec["user_filter"]), statements, exact[user])
        for user in SHARES:
            run(variant, "document_id filter + indexes, re-sorted", "iterative", user,
                resorted_sql(spec["user_filter"]), RELAXED, exact[user])
        for user in SHARES:
            statements = FILTERED_EXACT if len(owned_ids[user]) <= ROUTE_MAX_ROWS else RELAXED
            run(variant, "document_id filter, routed by rows owned", "routed", user,
                search_sql(spec["user_filter"]), statements, exact[user])

        if variant == "single owner":
            set_indexes(backend, [f"CREATE INDEX guide_sections_owner ON {SECTIONS} (owner)"])
            for user in SHARES:
                run(variant, "owner copied onto sections + index", "iterative", user,
                    search_sql("owner = auth.uid()"), RELAXED, exact[user])

    set_indexes(backend, [])
    if any(r["leaked_rows"] for r in rows):
        raise SystemExit("A user received rows they don't own")
    log("done")


if __name__ == "__main__":
    main()
