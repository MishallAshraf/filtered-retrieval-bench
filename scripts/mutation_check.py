"""Break the code on purpose and confirm the tests notice.

A test that has never failed may not be testing anything. Each mutation changes a copy of the
code, in a temporary folder, and the test suite must then fail.

    python scripts/mutation_check.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    ("nDCG ignores the ideal ranking", "src/metrics.py", "return dcg / ideal", "return dcg"),
    ("recall ignores k", "src/metrics.py", "set(ranked[:k]) & set(relevant)", "set(ranked) & set(relevant)"),
    ("MRR always credits the first hit as rank 1", "src/metrics.py", "return 1 / (i + 1)", "return 1.0"),
    ("hybrid drops the keyword ranking", "src/searches.py", "+ coalesce(1.0 / ({rrf_k} + lex.rnk), 0)", ""),
    ("keyword AND behaves like OR", "src/searches.py", '''return "plainto_tsquery('english', $3)"''',
     '''return "replace(plainto_tsquery('english', $3)::text, '&', '|')::tsquery"'''),
    ("minimum keyword score is ignored", "src/searches.py", "scored WHERE score >= {float(min_score)} ORDER BY score DESC, id LIMIT $2",
     "scored ORDER BY score DESC, id LIMIT $2"),
    ("RLS is never enabled", "src/db.py", 'conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")', "pass"),
    ("searches run as the table owner", "src/db.py", "SET LOCAL ROLE authenticated;", ""),
    ("plan check never reports HNSW", "src/db.py", "THEN 'hnsw'", "THEN 'seq'"),
    ("two users can claim one document", "src/ownership.py",
     'if not docs or any(owners[d] not in ("other", user) for d in docs):', "if not docs:"),
    ("a question can overfill a share", "src/ownership.py", "if len(new) > quota[user]:", "if False:"),
    ("gate ignores leaked rows", "src/gate.py", "if leaks(relevance) or leaks(fixes):", "if False:"),
    ("gate ignores lower scores", "src/gate.py", "elif current[name] < expected - tolerance:", "elif False:"),
    ("gate compares different setups", "src/gate.py",
     'if baseline["relevance_setup"] != setup_of(relevance) or baseline["fixes_setup"] != setup_of(fixes):', "if False:"),
    ("gate accepts an empty run", "src/gate.py", 'if not results["rows"]:', "if False:"),
]


def run_tests(folder):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
                          cwd=folder, capture_output=True, text=True).returncode


def main():
    survived = []
    for name, path, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            folder = Path(tmp) / "copy"
            folder.mkdir()
            for item in ("src", "tests"):
                shutil.copytree(ROOT / item, folder / item, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy(ROOT / "pytest.ini", folder)
            target = folder / path
            source = target.read_text(encoding="utf-8")
            if source.count(old) != 1:
                raise SystemExit(f"{name}: the text to change appears {source.count(old)} times")
            target.write_text(source.replace(old, new), encoding="utf-8")
            caught = run_tests(folder) != 0
        print(f"{'caught  ' if caught else 'SURVIVED'} {name}", flush=True)
        if not caught:
            survived.append(name)
    print(f"{len(MUTATIONS) - len(survived)} of {len(MUTATIONS)} caught")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
