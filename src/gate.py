"""Fail when results get worse than the saved baseline.

    python src/gate.py results/relevance_local.json results/fixes_local.json

The baseline records what it measured. A mismatch stops the check, because comparing different
setups would say nothing. A results file with no questions fails too: an empty run isn't a pass.
"""
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent / "baseline.json"
TOLERANCE = 0.01


def relevance_scores(results):
    """Results that fall are bad. The method and user scores come from test questions only."""
    scores = {}
    for method, entry in results["report"].items():
        if not entry["all"]["n"]:
            raise ValueError(f"{method} scored no questions")
        scores[f"{method} nDCG@10"] = entry["all"]["ndcg_at_10"]
        for user, summary in entry["by_user"].items():
            scores[f"{method} {user} nDCG@10"] = summary["ndcg_at_10"]
    return scores


def fix_scores(results):
    if not results["rows"]:
        raise ValueError("no fix results")
    return {f"{r['config']} / {r['setting']} / {r['user']} recall": r["recall_at_10"] for r in results["rows"]}


def setup_of(results):
    return {key: results.get(key) for key in ("dataset", "documents", "k", "model", "seeds", "seed", "route_max_rows")}


def leaks(results):
    if "rows" in results:
        return sum(r["leaked_rows"] for r in results["rows"])
    return sum(e["all"]["leaked_rows"] for e in results["report"].values())


def build(relevance, fixes):
    return dict(relevance_setup=setup_of(relevance), fixes_setup=setup_of(fixes),
                scores={**relevance_scores(relevance), **fix_scores(fixes)})


def check(baseline, relevance, fixes, tolerance=TOLERANCE):
    """Return a list of problems. An empty list passes."""
    problems = []
    if leaks(relevance) or leaks(fixes):
        problems.append("a user received rows they don't own")
    if baseline["relevance_setup"] != setup_of(relevance) or baseline["fixes_setup"] != setup_of(fixes):
        problems.append("results were measured with a different setup from the baseline")
        return problems
    current = {**relevance_scores(relevance), **fix_scores(fixes)}
    for name, expected in baseline["scores"].items():
        if name not in current:
            problems.append(f"{name}: missing from the results")
        elif current[name] < expected - tolerance:
            problems.append(f"{name}: {current[name]:.3f} is below the baseline {expected:.3f}")
    return problems


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: gate.py RELEVANCE_JSON FIXES_JSON [--record]")
    relevance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    fixes = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if "--record" in sys.argv:
        BASELINE.write_text(json.dumps(build(relevance, fixes), indent=2), encoding="utf-8")
        print(f"recorded {BASELINE.name}")
        return 0
    if not BASELINE.is_file():
        raise SystemExit("baseline.json is missing. Record one deliberately with --record.")
    problems = check(json.loads(BASELINE.read_text(encoding="utf-8")), relevance, fixes)
    for problem in problems:
        print("FAIL", problem)
    print("gate failed" if problems else "gate passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
