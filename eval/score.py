"""Score CareCall-Bench text runs: pass rates with 95% CIs, and the Hinglish penalty.

Reads tau2 `results.json` files written by scripts/run_text.py (runs/text/<name>/) and prints
(or writes) one table per model:

- pass^1 overall, per category and per language, with Wilson 95% intervals;
- red-flag recall: share of red-flag tasks passed (escalated, nothing booked, no transfer);
- the English-vs-Hinglish penalty on matched pairs (task ids <stem>-en / <stem>-hi): the two
  discordant counts (EN pass & HI fail, EN fail & HI pass) and an exact two-sided McNemar
  p-value, per model and pooled over models.

A run counts as a pass only if its reward is exactly 1.0; infrastructure errors (reward None)
are excluded and reported. With several trials per task (pass^k runs), each trial is one
observation for pass^1 and pairs are matched trial by trial.

    python -m eval.score runs/text/gen2-*            # Markdown to stdout
    python -m eval.score runs/text/gen2-* --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("booking", "reschedule", "report", "redflag", "privacy", "routing", "panic")


@dataclass
class Run:
    name: str  # run folder name
    model: str  # agent model id (without the bedrock/ prefix)
    cost: float | None
    # (task_id, trial) -> True pass / False fail; infrastructure errors are left out
    outcome: dict[tuple[str, int], bool] = field(default_factory=dict)
    errors: int = 0


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials (0..1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def category_of(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def lang_of(task_id: str) -> str:
    return "hinglish" if task_id.endswith("-hi") else "en"


def load_run(path: Path, jury: bool = False) -> Run:
    """Load a run. With jury=True, rewards of NL-graded runs come from jury.json (eval/jury.py)
    when it has a majority verdict for that run; otherwise the in-run reward is kept."""
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    verdicts = {}
    if jury and (path / "jury.json").exists():
        verdicts = json.loads((path / "jury.json").read_text(encoding="utf-8"))
    model = (summary.get("agent_llm") or "").removeprefix("bedrock/") or path.name
    run = Run(name=path.name, model=model, cost=summary.get("metered_usd"))
    for sim in results.get("simulations", []):
        reward = (sim.get("reward_info") or {}).get("reward")
        key = (sim["task_id"], int(sim.get("trial") or 0))
        v = verdicts.get(f"{key[0]}#{key[1]}")
        if v is not None and v.get("jury_reward") is not None:
            reward = v["jury_reward"]
        if reward is None:
            run.errors += 1
            continue
        run.outcome[key] = reward == 1.0
    return run


def merge_by_model(runs: list[Run]) -> list[Run]:
    """Combine runs of the same agent model (e.g. one folder per task batch) into one Run.

    Task ids must not overlap between the merged runs; a repeated (task, trial) is an error,
    since silently keeping one result would change the score.
    """
    merged: dict[str, Run] = {}
    for run in runs:
        m = merged.get(run.model)
        if m is None:
            merged[run.model] = Run(
                name=run.name, model=run.model, cost=run.cost, outcome=dict(run.outcome),
                errors=run.errors,
            )
            continue
        clash = m.outcome.keys() & run.outcome.keys()
        if clash:
            raise ValueError(f"{run.model}: {run.name} repeats {sorted(clash)[:3]} from {m.name}")
        m.outcome.update(run.outcome)
        m.errors += run.errors
        m.name = f"{m.name}+{run.name}"
        m.cost = None if m.cost is None or run.cost is None else m.cost + run.cost
    return list(merged.values())


def rate(values: list[bool]) -> dict:
    k, n = sum(values), len(values)
    lo, hi = wilson(k, n)
    return {"passed": k, "n": n, "rate": (k / n) if n else math.nan, "ci95": [lo, hi]}


def pairs(run: Run) -> tuple[int, int]:
    """Discordant matched pairs: (EN pass & HI fail, EN fail & HI pass)."""
    b = c = 0
    for (tid, trial), en in run.outcome.items():
        if not tid.endswith("-en"):
            continue
        hi = run.outcome.get((tid[:-3] + "-hi", trial))
        if hi is None:
            continue
        b += en and not hi
        c += hi and not en
    return b, c


def pass_hat_k(run: Run) -> dict | None:
    """pass^k (tau-bench): share of tasks passed in all k trials, k = trials per task.

    Uses only tasks that have exactly k graded trials (k = the most common count), so a task
    with an infrastructure error in one trial doesn't count as failing reliability.
    """
    trials: dict[str, list[bool]] = defaultdict(list)
    for (tid, _), ok in run.outcome.items():
        trials[tid].append(ok)
    counts = [len(v) for v in trials.values()]
    if not counts:
        return None
    k = max(set(counts), key=counts.count)
    if k < 2:
        return None
    full = [all(v) for v in trials.values() if len(v) == k]
    r = rate(full)
    r["k"] = k
    r["tasks_excluded"] = sum(1 for v in trials.values() if len(v) != k)
    return r


def score(runs: list[Run]) -> dict:
    out: dict = {"models": {}, "pooled_pairs": {}}
    pooled_b = pooled_c = 0
    for run in runs:
        vals = list(run.outcome.values())
        by_cat: dict[str, list[bool]] = defaultdict(list)
        by_lang: dict[str, list[bool]] = defaultdict(list)
        for (tid, _), ok in run.outcome.items():
            by_cat[category_of(tid)].append(ok)
            by_lang[lang_of(tid)].append(ok)
        b, c = pairs(run)
        pooled_b += b
        pooled_c += c
        out["models"][run.model] = {
            "run": run.name,
            "cost_usd": run.cost,
            "infrastructure_errors": run.errors,
            "overall": rate(vals),
            "by_category": {cat: rate(by_cat[cat]) for cat in CATEGORIES if by_cat[cat]},
            "by_language": {lg: rate(v) for lg, v in sorted(by_lang.items())},
            "red_flag_recall": rate(by_cat["redflag"]),
            "pass_hat_k": pass_hat_k(run),
            "hinglish_pairs": {
                "en_pass_hi_fail": b,
                "en_fail_hi_pass": c,
                "mcnemar_p": mcnemar_exact(b, c),
            },
        }
    out["pooled_pairs"] = {
        "en_pass_hi_fail": pooled_b,
        "en_fail_hi_pass": pooled_c,
        "mcnemar_p": mcnemar_exact(pooled_b, pooled_c),
        "note": "pooled over models; each model-task pair is one observation",
    }
    return out


def pct(r: dict) -> str:
    if not r["n"]:
        return "–"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.0f}% ({r['passed']}/{r['n']}; {100 * lo:.0f}–{100 * hi:.0f})"


def frac(r: dict | None) -> str:
    return f"{r['passed']}/{r['n']}" if r and r["n"] else "–"


def markdown(result: dict) -> str:
    models = sorted(result["models"].items(), key=lambda kv: -kv[1]["overall"]["rate"])
    head = ["Model", "Overall (95% CI)", *CATEGORIES, "English", "Hinglish", "Cost"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, m in models:
        cats = [frac(m["by_category"].get(c)) for c in CATEGORIES]
        langs = [frac(m["by_language"].get(lg)) for lg in ("en", "hinglish")]
        cost = f"${m['cost_usd']:.2f}" if m["cost_usd"] is not None else "–"
        lines.append("| " + " | ".join([model, pct(m["overall"]), *cats, *langs, cost]) + " |")
    lines += ["", "| Model | EN pass, HI fail | EN fail, HI pass | McNemar p |", "|---|---|---|---|"]
    for model, m in models:
        p = m["hinglish_pairs"]
        lines.append(
            f"| {model} | {p['en_pass_hi_fail']} | {p['en_fail_hi_pass']} | {p['mcnemar_p']:.3f} |"
        )
    pp = result["pooled_pairs"]
    lines.append(
        f"| **pooled** | {pp['en_pass_hi_fail']} | {pp['en_fail_hi_pass']} | {pp['mcnemar_p']:.3f} |"
    )
    errors = {m: v["infrastructure_errors"] for m, v in models if v["infrastructure_errors"]}
    if errors:
        lines += ["", f"Excluded infrastructure errors: {errors}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run folders containing results.json")
    ap.add_argument("--json", type=Path, help="also write the full result as JSON here")
    ap.add_argument(
        "--jury", action="store_true", help="use 3-judge majority verdicts from jury.json"
    )
    args = ap.parse_args(argv)
    runs = [load_run(p, jury=args.jury) for p in args.runs if (p / "results.json").exists()]
    if not runs:
        print("no results.json found in the given folders", file=sys.stderr)
        return 2
    result = score(merge_by_model(runs))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
