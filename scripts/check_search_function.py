"""Check the recommended search function on the guide's schema, beyond results per user.

1. It is SECURITY INVOKER, and its filter selects exactly the rows the RLS policy allows.
2. EXPLAIN ANALYZE shows small users on the btree indexes and large users on HNSW.
3. Users near the planner's switch point, and users whose documents are far from the
   questions, still get full, correct results.
4. One pooled connection serving several users can't carry one user's cached plan to another.
5. The function logs `search_shortfall` whenever it returns fewer than match_count results
   to a user who owns at least match_count sections.

Extra users are made by reassigning documents from `other`, and are handed back at the end.
The functions are dropped afterwards, so they aren't left exposed as API endpoints.

    python scripts/check_search_function.py --backend supabase
"""
import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db  # noqa: E402
import fiqa  # noqa: E402
from evaluate_fixes import EXACT, K  # noqa: E402
from evaluate_guide_schema import DOCUMENTS, SECTIONS, VARIANTS, set_indexes, search_sql  # noqa: E402
from metrics import recall_against_exact  # noqa: E402
from ownership import SEED, SHARES  # noqa: E402

FUNCTION = "public.match_guide_sections"
BROKEN = "public.match_guide_sections_broken"
LOOKUP = f"SELECT d.id FROM {DOCUMENTS} d WHERE d.owner_id = auth.uid()"
SPEC = VARIANTS["single owner"]
NEAREST = (f"WITH nearest AS MATERIALIZED (SELECT s.id, s.embedding <=> {{vector}} AS distance FROM {SECTIONS} s "
           f"WHERE s.document_id IN ({LOOKUP}) ORDER BY distance LIMIT {{limit}}) "
           "SELECT nearest.id, nearest.distance FROM nearest ORDER BY nearest.distance + 0")
UNFILTERED = (f"SELECT s.id, s.embedding <=> query_embedding AS distance FROM {SECTIONS} s "
              "ORDER BY s.embedding <=> query_embedding LIMIT match_count")
# Documents taken from `other`. Far users get the documents least similar to any test question.
FAR_USERS = {"far_0_5pct": 36, "far_3pct": 216, "far_10pct": 720}
SWITCH_USERS = {"sw_1_5pct": 108, "sw_2pct": 144, "sw_2_5pct": 180, "sw_3pct": 216, "sw_4pct": 288}
PLAN_SAMPLE = 20


def function_sql(name, query, iterative=True):
    setting = "SET hnsw.iterative_scan = relaxed_order" if iterative else ""
    return f"""CREATE OR REPLACE FUNCTION {name}(query_embedding vector(384), match_count int)
RETURNS TABLE (id int, distance float8) LANGUAGE plpgsql STABLE {setting} AS $$
DECLARE
  returned int;
  owned int;
BEGIN
  RETURN QUERY {query};
  GET DIAGNOSTICS returned = ROW_COUNT;
  -- Only a short result pays for the check, and the count stops at match_count.
  IF returned < match_count THEN
    SELECT count(*) INTO owned FROM (
      SELECT 1 FROM {SECTIONS} s WHERE s.document_id IN ({LOOKUP}) LIMIT match_count) o;
    IF owned >= match_count THEN
      RAISE LOG 'search_shortfall user=% returned=% requested=% owned_at_least=%',
        auth.uid(), returned, match_count, owned;
    END IF;
  END IF;
END $$"""


class Session:
    """One connection reused across users, as a connection pooler does, one transaction per request."""

    def __init__(self, backend):
        self.conn = db.connect(backend)
        self.conn.commit()
        self.conn.autocommit = True
        self.notices = []
        self.conn.add_notice_handler(lambda d: self.notices.append(d.message_primary))

    def run(self, user, sql, params=(), settings=()):
        with self.conn.transaction():
            self.conn.execute("SELECT set_config('request.jwt.claims', %s, true)",
                              (json.dumps({"sub": str(db.user_id(user)), "role": "authenticated"}),))
            self.conn.execute("SET LOCAL ROLE authenticated")
            self.conn.execute("SET LOCAL client_min_messages = log")
            for statement in settings:
                self.conn.execute(statement)
            return self.conn.execute(sql, params).fetchall()

    def close(self):
        self.conn.close()


def assign_extra_users(backend, vectors, query_vectors, log):
    """Reassign some of `other`'s documents to the far and switch-point users."""
    with db.connect(backend) as conn:
        rows = conn.execute(f"SELECT s.id, s.document_id FROM {SECTIONS} s JOIN {DOCUMENTS} d ON d.id = s.document_id "
                            "WHERE d.owner_id = %s", (db.user_id("other"),)).fetchall()
        closest = vectors[[r[0] for r in rows]] @ query_vectors.T
        best_by_document = {}
        for (section, document), score in zip(rows, closest.max(axis=1)):
            best_by_document[document] = max(best_by_document.get(document, -1.0), float(score))
        by_distance = sorted(best_by_document, key=best_by_document.get)
        start = 0
        for user, count in FAR_USERS.items():
            chosen = by_distance[start:start + count]
            start += count
            conn.execute(f"UPDATE {DOCUMENTS} SET owner_id = %s WHERE id = ANY(%s)", (db.user_id(user), chosen))
            log(f"{user}: {count} documents, best match to any test question {best_by_document[chosen[-1]]:.3f} or lower")
        rest = by_distance[start:]
        random.Random(SEED + 2).shuffle(rest)
        start = 0
        for user, count in SWITCH_USERS.items():
            conn.execute(f"UPDATE {DOCUMENTS} SET owner_id = %s WHERE id = ANY(%s)",
                         (db.user_id(user), rest[start:start + count]))
            start += count
        conn.execute(f"UPDATE {SECTIONS} s SET owner = d.owner_id FROM {DOCUMENTS} d WHERE d.id = s.document_id")
        conn.execute(f"ANALYZE {DOCUMENTS}")
        conn.execute(f"ANALYZE {SECTIONS}")


def restore_users(backend):
    extra = [db.user_id(u) for u in {**FAR_USERS, **SWITCH_USERS}]
    with db.connect(backend) as conn:
        conn.execute(f"UPDATE {DOCUMENTS} SET owner_id = %s WHERE owner_id = ANY(%s)", (db.user_id("other"), extra))
        conn.execute(f"UPDATE {SECTIONS} s SET owner = d.owner_id FROM {DOCUMENTS} d WHERE d.id = s.document_id")
        conn.execute(f"DROP FUNCTION IF EXISTS {FUNCTION}(vector, int)")
        conn.execute(f"DROP FUNCTION IF EXISTS {BROKEN}(vector, int)")
        conn.execute(f"ANALYZE {DOCUMENTS}")
        conn.execute(f"ANALYZE {SECTIONS}")
    set_indexes(backend, [])


def owned_sections(backend):
    with db.connect(backend) as conn:
        rows = conn.execute(f"SELECT d.owner_id, s.id FROM {SECTIONS} s JOIN {DOCUMENTS} d ON d.id = s.document_id").fetchall()
    owned = {}
    for owner, section in rows:
        owned.setdefault(owner, set()).add(section)
    return owned


def check_definition(backend, session, users, owned):
    with db.connect(backend) as conn:
        definer = conn.execute("SELECT prosecdef FROM pg_proc WHERE oid = %s::regprocedure",
                               (f"{FUNCTION}(vector, int)",)).fetchone()[0]
        policy = conn.execute("SELECT qual FROM pg_policies WHERE schemaname = 'public' AND tablename = 'guide_sections' "
                              "AND policyname = 'own_rows'").fetchone()[0]
        body = conn.execute("SELECT pg_get_functiondef(%s::regprocedure)", (f"{FUNCTION}(vector, int)",)).fetchone()[0]
        function_filter = re.search(r"WHERE (s\.document_id IN \(.*?auth\.uid\(\)\))", body).group(1)
    # The same rows? Compare what the policy shows each user with what the function's
    # filter selects for them with RLS out of the way.
    mismatched = {}
    for user in users:
        visible = {r[0] for r in session.run(user, f"SELECT id FROM {SECTIONS}")}
        filtered = owned.get(db.user_id(user), set())
        if visible != filtered:
            mismatched[user] = len(visible ^ filtered)
    return dict(security_definer=definer, policy=policy, function_filter=function_filter, users_checked=len(users),
                users_with_different_rows=mismatched)


def explain(session, user, vector):
    # TIMING OFF: per-node clock reads cost up to ~1 s per search on the hosted project's
    # VM, mostly on HNSW scans, while the search itself takes a few ms.
    sql = "EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) " + NEAREST.format(vector="%s", limit="10")
    rows = session.run(user, sql, (vector,), ["SET LOCAL hnsw.iterative_scan = relaxed_order"])
    text = "\n".join(r[0] for r in rows)
    path = "+".join(p for p, pattern in (("hnsw", "guide_sections_embedding"), ("docid", "guide_sections_docid"),
                                          ("owner", "guide_documents_owner")) if pattern in text) or "seq"
    ms = float(re.search(r"Execution Time: ([\d.]+) ms", text).group(1))
    return path, ms, text


PGSS = ("SELECT coalesce(sum(calls), 0), coalesce(sum(total_exec_time), 0) FROM extensions.pg_stat_statements "
        "WHERE query LIKE 'WITH nearest AS MATERIALIZED%%'")


def _stats(values):
    values = list(values)
    return dict(n=len(values), p50=statistics.median(values), mean=statistics.fmean(values), max=max(values))


def time_searches(backend, session, user, query_vectors, qids):
    """Time each search with EXPLAIN (ANALYZE, TIMING OFF) and with pg_stat_statements.

    On the hosted free-tier project the first run of a new query vector takes 0.3-1.8 s
    however it is timed, while repeats take a few ms, so first runs and repeats are kept
    apart. The method that runs first alternates between searches, so both methods time
    first runs and repeats. The first search of the run is warm-up and is reported apart."""
    rows = []
    with db.connect(backend) as admin:
        admin.commit()
        admin.autocommit = True

        def plain(vector):
            calls0, total0 = admin.execute(PGSS).fetchone()
            session.run(user, NEAREST.format(vector="%s", limit="10"), (vector,),
                        ["SET LOCAL hnsw.iterative_scan = relaxed_order"])
            calls1, total1 = admin.execute(PGSS).fetchone()
            if calls1 - calls0 != 1:
                raise RuntimeError(f"pg_stat_statements saw {calls1 - calls0} calls for one search")
            return float(total1 - total0)

        for i, q in enumerate(qids):
            vector = query_vectors[q]
            order = ["explain", "pgss"] if i % 2 == 0 else ["pgss", "explain"]
            times, path = {}, None
            for run in ("first", "repeat"):
                for method in order:
                    if method == "explain":
                        path, ms, _ = explain(session, user, vector)
                    else:
                        ms = plain(vector)
                    times.setdefault(method, []).append(ms)
            rows.append(dict(qid=q, plan=path, first_method=order[0], first_ms=times[order[0]][0],
                             explain_repeat_ms=times["explain"][-1], pgss_repeat_ms=times["pgss"][-1]))
    warm, rest = rows[0], rows[1:]
    return dict(
        searches=len(rest), warm_up=warm,
        first_run_explain_ms=_stats(r["first_ms"] for r in rest if r["first_method"] == "explain"),
        first_run_pgss_ms=_stats(r["first_ms"] for r in rest if r["first_method"] == "pgss"),
        repeat_explain_ms=_stats(r["explain_repeat_ms"] for r in rest),
        repeat_pgss_ms=_stats(r["pgss_repeat_ms"] for r in rest),
        plans={p: sum(r["plan"] == p for r in rest) for p in {r["plan"] for r in rest}},
        rows=rows)


def _jwt(secret, sub):
    """A short-lived HS256 token for an existing test user id, like the one Supabase Auth issues."""
    import base64, hashlib, hmac
    enc = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    now = int(time.time())
    head = enc(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = enc(json.dumps({"sub": sub, "role": "authenticated", "aud": "authenticated",
                           "iat": now, "exp": now + 600}).encode())
    sig = enc(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def time_rpc(backend, user, query_vectors, qids):
    """Time searches through PostgREST, as supabase.rpc() calls them, from this machine.

    Needs SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_JWT_SECRET (legacy JWT secret) in .env;
    returns None without them. The first call also opens the TLS connection and is reported apart."""
    import requests
    env = dict(line.split("=", 1) for line in (db.ROOT / ".env").read_text(encoding="utf-8-sig").splitlines()
               if "=" in line and not line.lstrip().startswith("#"))
    url, key, secret = (env.get(k, "").strip() for k in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET"))
    if not (url and key and secret):
        return None
    with db.connect(backend) as conn:
        conn.execute("NOTIFY pgrst, 'reload schema'")
        conn.commit()
    time.sleep(3)
    http = requests.Session()
    http.headers.update(apikey=key, Authorization=f"Bearer {_jwt(secret, str(db.user_id(user)))}")
    rows = []
    for q in qids:
        body = dict(query_embedding="[" + ",".join(f"{x:.7f}" for x in query_vectors[q]) + "]", match_count=10)
        row = dict(qid=q)
        for run in ("first_ms", "repeat_ms"):
            started = time.perf_counter()
            r = http.post(f"{url.rstrip('/')}/rest/v1/rpc/{FUNCTION.split('.')[-1]}", json=body, timeout=30)
            row[run] = (time.perf_counter() - started) * 1000
            if not r.ok:
                raise RuntimeError(f"rpc returned {r.status_code}: {r.text[:300]}")
            row["returned"] = len(r.json())
        rows.append(row)
    warm, rest = rows[0], rows[1:]
    return dict(user=user, searches=len(rest), warm_up=warm, first_run_ms=_stats(r["first_ms"] for r in rest),
                repeat_ms=_stats(r["repeat_ms"] for r in rest), returned=sorted({r["returned"] for r in rest}),
                rows=rows)


def run_function(session, user, name, query_vectors, qids):
    """Call the function as the user for each question. Returns ids per question and alerts."""
    found, alerts = {}, 0
    for qid in qids:
        before = len(session.notices)
        found[qid] = [r[0] for r in session.run(user, f"SELECT id FROM {name}(%s, %s)", (query_vectors[qid], K))]
        alerts += sum("search_shortfall" in n for n in session.notices[before:])
    return found, alerts


def summarise(found, exact, owned_ids, vectors, query_vectors, alerts):
    returned = [len(ids) for ids in found.values()]
    unordered = sum(bool(np.any(np.diff(1 - vectors[ids] @ query_vectors[q]) < -1e-6)) for q, ids in found.items() if len(ids) > 1)
    short = sum(n < K for n in returned)
    return dict(mean_returned=statistics.mean(returned), short_results=short,
                recall_at_10=statistics.mean(recall_against_exact(found[q], exact[q]["ids"]) for q in found),
                out_of_order=unordered, leaked_rows=sum(len(set(ids) - owned_ids) for ids in found.values()),
                alerts=alerts, missed_alerts=short - alerts if len(owned_ids) >= K else 0)


LARGE_USERS = ["u_50pct", "u_10pct", "far_10pct"]


def retime_large_users(backend, session, out, log, query_vectors, test_qids):
    """Replace the large users' timings, which EXPLAIN's per-node timing inflated, in the saved results."""
    report = json.loads(out.read_text(encoding="utf-8"))
    qids = test_qids[:PLAN_SAMPLE + 1]
    log("re-timing large users: EXPLAIN (ANALYZE, TIMING OFF) and pg_stat_statements; "
        "first runs of each query vector and repeats kept apart; first search of each run is warm-up")
    f = lambda t: f"p50={t['p50']:.1f} mean={t['mean']:.1f} max={t['max']:.1f}ms"
    for user in LARGE_USERS:
        t = time_searches(backend, session, user, query_vectors, qids)
        row = report["users"][user]
        row["explain_p50_ms_timing_on"] = row.pop("explain_p50_ms_timing_on", row["explain_p50_ms"])
        row["explain_p50_ms"] = t["repeat_explain_ms"]["p50"]
        row["timing"] = t
        w = t["warm_up"]
        log(f"{user:11} plans={t['plans']} | first run: explain {f(t['first_run_explain_ms'])}, "
            f"pgss {f(t['first_run_pgss_ms'])} | repeat: explain {f(t['repeat_explain_ms'])}, "
            f"pgss {f(t['repeat_pgss_ms'])} | warm-up {w['first_method']} {w['first_ms']:.1f}ms "
            f"(TIMING ON before: {row['explain_p50_ms_timing_on']:.1f}ms)")
    report["timing_method"] = {
        "large_users": f"{PLAN_SAMPLE} searches each after one warm-up. Each query vector runs twice with "
                       "EXPLAIN (ANALYZE, TIMING OFF) and twice plainly, timed by pg_stat_statements. The first "
                       "run of a new vector is slow on this free-tier project however it is timed; "
                       "explain_p50_ms is the repeat.",
        "other_users": "EXPLAIN (ANALYZE, BUFFERS) with per-node timing, first run of each vector; "
                       "their p50s are upper bounds"}
    rpc = time_rpc(backend, "u_10pct", query_vectors, qids)
    if rpc is None:
        log("rpc timing skipped: SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_JWT_SECRET are not all in .env")
    else:
        report["rpc_latency"] = rpc
        log(f"rpc (PostgREST from this machine, {rpc['user']}, returned={rpc['returned']}): "
            f"first run {f(rpc['first_run_ms'])} | repeat {f(rpc['repeat_ms'])} | "
            f"warm-up (opens TLS) {rpc['warm_up']['first_ms']:.0f}ms")
    db.write_json(out, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "supabase"], required=True)
    parser.add_argument("--timing-only", action="store_true",
                        help="re-time the large users only and merge the timings into the existing results")
    args = parser.parse_args()
    backend = args.backend
    out = db.ROOT / "results" / f"function_checks_{backend}.json"
    logfile = open(db.ROOT / "results" / f"function_checks_{backend}.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        logfile.write(message + "\n")
        logfile.flush()

    vectors, _, _ = fiqa.documents()
    questions = fiqa.questions()
    test_qids = [q[0] for q in questions if q[1] == "test"]
    query_vectors = {q[0]: q[3] for q in questions}
    users = list(SHARES) + list(FAR_USERS) + list(SWITCH_USERS)
    report = dict(questions=len(test_qids), k=K)

    set_indexes(backend, [f"CREATE INDEX guide_sections_docid ON {SECTIONS} (document_id)", SPEC["lookup_index"]])
    db.set_policy(backend, SECTIONS, SPEC["policy"])
    with db.connect(backend) as conn:
        conn.execute(function_sql(FUNCTION, NEAREST.format(vector="query_embedding", limit="match_count")))
        conn.execute(function_sql(BROKEN, UNFILTERED, iterative=False))
        for name in (FUNCTION, BROKEN):
            conn.execute(f"GRANT EXECUTE ON FUNCTION {name}(vector, int) TO authenticated")
    session = Session(backend)
    try:
        assign_extra_users(backend, vectors, np.stack([query_vectors[q] for q in test_qids]), log)
        owned = owned_sections(backend)
        owned_ids = {u: owned.get(db.user_id(u), set()) for u in users}
        if args.timing_only:
            retime_large_users(backend, session, out, log, query_vectors, test_qids)
            return

        report["definition"] = check_definition(backend, session, users, owned)
        log(f"definition: {json.dumps(report['definition'])}")

        report["users"] = {}
        for user in users:
            started = time.time()
            exact = db.search(backend, user, search_sql(), EXACT, K, test_qids)
            plans = [explain(session, user, query_vectors[q]) for q in test_qids[:PLAN_SAMPLE]]
            found, alerts = run_function(session, user, FUNCTION, query_vectors, test_qids)
            row = dict(owned_sections=len(owned_ids[user]),
                       plans={p: sum(x[0] == p for x in plans) for p in {x[0] for x in plans}},
                       explain_p50_ms=statistics.median(x[1] for x in plans),
                       **summarise(found, exact, owned_ids[user], vectors, query_vectors, alerts))
            report["users"][user] = row
            report.setdefault("sample_plans", {})[user] = plans[0][2]
            log(f"{user:11} owns={row['owned_sections']:6} plans={row['plans']} p50={row['explain_p50_ms']:.1f}ms "
                f"returned={row['mean_returned']:.2f} short={row['short_results']} recall={row['recall_at_10']:.3f} "
                f"unordered={row['out_of_order']} leaked={row['leaked_rows']} alerts={row['alerts']} "
                f"missed_alerts={row['missed_alerts']} ({time.time() - started:.0f}s)")
            db.write_json(out, report)

        # A pooled connection: the large user warms the cache, then the small user follows on
        # the same connection, as PostgREST would schedule them.
        qids = test_qids[:100]
        small, large = "u_0_1pct", "u_50pct"
        exact_small = db.search(backend, small, search_sql(), EXACT, K, qids)
        mixed = Session(backend)
        try:
            run_function(mixed, large, FUNCTION, query_vectors, qids[:20])
            found, alerts = run_function(mixed, small, FUNCTION, query_vectors, qids)
            prepared_plans = []
            mixed.conn.execute("PREPARE nearest(vector, int) AS " + NEAREST.format(vector="$1", limit="$2"))
            for user in [large] * 8 + [small] * 3:
                literal = "[" + ",".join(f"{x:.7f}" for x in query_vectors[qids[0]]) + "]"
                text = "\n".join(r[0] for r in mixed.run(
                    user, f"EXPLAIN EXECUTE nearest('{literal}'::vector, 10)", (),
                    ["SET LOCAL hnsw.iterative_scan = relaxed_order"]))
                prepared_plans.append((user, "hnsw" if "guide_sections_embedding" in text else "btree"))
        finally:
            mixed.close()
        report["pooled_connection"] = dict(
            small_after_large=summarise(found, exact_small, owned_ids[small], vectors, query_vectors, alerts),
            prepared_statement_plans=prepared_plans)
        log(f"pooled connection: {json.dumps(report['pooled_connection'])}")

        # The alert must fire when results really are short: the unfiltered function without
        # iterative scan is the original bug.
        exact_1 = db.search(backend, "u_1pct", search_sql(), EXACT, K, qids)
        found, alerts = run_function(session, "u_1pct", BROKEN, query_vectors, qids)
        report["alert_control"] = summarise(found, exact_1, owned_ids["u_1pct"], vectors, query_vectors, alerts)
        report["alert_sample"] = next((n for n in session.notices if "search_shortfall" in n), None)
        log(f"alert control: {json.dumps(report['alert_control'])}; sample: {report['alert_sample']}")
        db.write_json(out, report)
    finally:
        session.close()
        restore_users(backend)
        log("restored owners, dropped functions and optional indexes")


if __name__ == "__main__":
    main()
