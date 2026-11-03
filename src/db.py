"""Connections, schema and the one function every measured search goes through.

The same code runs against local Postgres and a hosted Supabase project. Locally, auth.uid()
and the authenticated role are recreated with Supabase's definition, so both use the same
RLS policies. Searches run inside the database, so latency excludes the network trip.
"""
import json
import re
import time
import uuid
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
LOCAL_PGDATA = ROOT / "data" / "pgdata_bench"
NAMESPACE = uuid.UUID("6f1d7d3e-52a7-4b49-9a55-0e7a1c2b9d10")
BATCH = 2000

WRAPPED_POLICY = "owner = (select auth.uid())"
PLAIN_POLICY = "owner = auth.uid()"
SAFE_TABLE = re.compile(r"^public\.[a-z_]+$")

LOCAL_AUTH = """
CREATE SCHEMA IF NOT EXISTS auth;
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS $$
  SELECT coalesce(nullif(current_setting('request.jwt.claim.sub', true), ''),
                  (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub'))::uuid
$$;
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated;
  END IF;
END $$;
GRANT USAGE ON SCHEMA auth TO authenticated;
"""

# The plan is read with EXPLAIN on the exact query and parameters that are then run, for every
# question, because the planner can choose differently from one question to the next.
SEARCH_FUNCTION = r"""
CREATE OR REPLACE FUNCTION bench.search(user_id uuid, k int, search_sql text, qid_filter int[] DEFAULT NULL)
RETURNS TABLE (qid int, ids int[], ms double precision, plan text) LANGUAGE plpgsql AS $$
DECLARE
  qids int[];
  vectors vector[];
  questions text[];
  started timestamptz;
  line text;
  explained text;
  wrapped text := 'SELECT coalesce(array_agg(id), ''{}'') FROM (' || search_sql || ') s';
BEGIN
  SELECT array_agg(q.qid ORDER BY q.qid), array_agg(q.embedding ORDER BY q.qid),
         array_agg(q.question ORDER BY q.qid)
    INTO qids, vectors, questions FROM bench.questions q
    WHERE qid_filter IS NULL OR q.qid = ANY(qid_filter);
  PERFORM set_config('request.jwt.claims',
                     json_build_object('sub', user_id, 'role', 'authenticated')::text, true);
  SET LOCAL ROLE authenticated;
  FOR i IN 1..coalesce(array_length(qids, 1), 0) LOOP
    explained := '';
    FOR line IN EXECUTE 'EXPLAIN ' || wrapped USING vectors[i], k, questions[i] LOOP
      explained := explained || line || E'\n';
    END LOOP;
    plan := concat_ws('+',
      CASE WHEN explained ~ '(using|on) \w+_embedding' THEN 'hnsw' END,
      CASE WHEN explained ~ '(using|on) \w+_owner' THEN 'owner' END,
      CASE WHEN explained ~ '(using|on) \w+_docid' THEN 'docid' END,
      CASE WHEN explained ~ '(using|on) \w+_tsv' THEN 'gin' END,
      CASE WHEN explained ~ 'Seq Scan' THEN 'seq' END);
    started := clock_timestamp();
    EXECUTE wrapped INTO ids USING vectors[i], k, questions[i];
    ms := extract(epoch FROM clock_timestamp() - started) * 1000;
    qid := qids[i];
    RETURN NEXT;
  END LOOP;
END $$
"""

_server = None


def user_id(user):
    return uuid.uuid5(NAMESPACE, user)


def _url(backend):
    global _server
    if backend == "local":
        import pixeltable_pgserver
        if _server is None:
            _server = pixeltable_pgserver.get_server(LOCAL_PGDATA, cleanup_mode="stop")
        return _server.get_uri()
    if backend == "supabase":
        for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("SUPABASE_DB_URL="):
                return line.split("=", 1)[1].strip()
        raise SystemExit("SUPABASE_DB_URL is missing from .env")
    raise ValueError(f"unknown backend {backend}")


def connect(backend):
    conn = psycopg.connect(_url(backend), keepalives=1, keepalives_idle=30)
    conn.execute("SET statement_timeout = 0")
    conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if conn.execute("SELECT count(*) FROM pg_type WHERE typname = 'vector'").fetchone()[0]:
        register_vector(conn)
    return conn


def setup(backend):
    with connect(backend) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if backend == "local":
            conn.execute(LOCAL_AUTH)
        conn.commit()
    with connect(backend) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS bench")
        conn.execute("REVOKE ALL ON SCHEMA bench FROM PUBLIC")
        conn.execute("CREATE TABLE IF NOT EXISTS bench.questions (qid int PRIMARY KEY, split text NOT NULL, "
                     "question text NOT NULL, embedding vector(384) NOT NULL)")
        conn.execute(SEARCH_FUNCTION)
        conn.execute("REVOKE ALL ON FUNCTION bench.search(uuid, int, text, int[]) FROM PUBLIC")
        conn.commit()


def load_questions(backend, rows):
    """rows are (qid, split, question, embedding)."""
    with connect(backend) as conn:
        if conn.execute("SELECT count(*) FROM bench.questions").fetchone()[0] == len(rows):
            return
        conn.execute("TRUNCATE bench.questions")
        with conn.cursor().copy("COPY bench.questions (qid, split, question, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "text", "text", "vector"])
            for row in rows:
                copy.write_row(row)
        conn.commit()


def load_docs(backend, table, owners, vectors, texts=None, log=print):
    """Create or resume a documents table with its HNSW index. Rows upload in committed
    batches so a dropped connection doesn't lose earlier work."""
    if not SAFE_TABLE.match(table):
        raise ValueError(f"unexpected table name {table}")
    name = table.split(".")[1]
    with connect(backend) as conn:
        extra = ", content text NOT NULL, content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED" if texts else ""
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id int PRIMARY KEY, owner uuid NOT NULL, embedding vector(384) NOT NULL{extra})")
        conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"GRANT SELECT ON {table} TO authenticated")
        conn.commit()
        loaded = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        conn.commit()
        columns = "id, owner, embedding, content" if texts else "id, owner, embedding"
        types = ["int4", "uuid", "vector", "text"] if texts else ["int4", "uuid", "vector"]
        started = time.time()
        for first in range(loaded, len(owners), BATCH):
            with conn.cursor().copy(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(types)
                for i in range(first, min(first + BATCH, len(owners))):
                    row = (i, user_id(owners[i]), vectors[i])
                    copy.write_row(row + (texts[i],) if texts else row)
            conn.commit()
        log(f"{table}: {len(owners)} rows ready, uploaded in {time.time() - started:.0f}s")

        started = time.time()
        # Nano compute has too little shared memory for a parallel in-memory build.
        conn.execute(f"SET maintenance_work_mem = '{'1GB' if backend == 'local' else '64MB'}'")
        if backend != "local":
            conn.execute("SET max_parallel_maintenance_workers = 0")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_embedding ON {table} USING hnsw (embedding vector_cosine_ops)")
        if texts:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_tsv ON {table} USING gin (content_tsv)")
        conn.execute(f"ANALYZE {table}")
        conn.commit()
        log(f"{table}: indexes ready in {time.time() - started:.0f}s; database "
            + conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0])
        conn.commit()


def set_policy(backend, table, using):
    with connect(backend) as conn:
        conn.execute(f"DROP POLICY IF EXISTS own_docs ON {table}")
        conn.execute(f"DROP POLICY IF EXISTS own_rows ON {table}")
        conn.execute(f"CREATE POLICY own_rows ON {table} FOR SELECT TO authenticated USING ({using})")


def set_owner_index(backend, table, present):
    name = table.split(".")[1]
    with connect(backend) as conn:
        if present:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_owner ON {table} (owner)")
        else:
            conn.execute(f"DROP INDEX IF EXISTS public.{name}_owner")
        conn.execute(f"ANALYZE {table}")


def search(backend, user, sql, settings=(), k=10, qids=None):
    """Run a search for every question as this user. Returns {qid: dict(ids, ms, plan)}."""
    with connect(backend) as conn:
        for statement in settings:
            conn.execute(statement)
        rows = conn.execute("SELECT qid, ids, ms, plan FROM bench.search(%s, %s, %s, %s) ORDER BY qid",
                            (user_id(user), k, sql, qids)).fetchall()
        conn.rollback()
    return {r[0]: dict(ids=r[1], ms=r[2], plan=r[3]) for r in rows}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
