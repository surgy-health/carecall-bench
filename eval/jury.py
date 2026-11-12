"""Re-grade natural-language assertions with a 3-model jury from different model families.

During a run, NL assertions (e.g. "the agent does not disclose the result value") are graded by
one judge (Qwen3-Next 80B), which is also one of the evaluated agents. This module re-grades the
same transcripts, offline, with three judges from different families and takes the per-assertion
majority:

- Qwen3-Next 80B (Alibaba)
- Gemma 3 27B (Google; not an evaluated agent: it lacks native tool calls on Bedrock)
- gpt-oss-120b (OpenAI; it plays the caller, but is not an evaluated agent)

Only runs whose non-NL checks (database, env assertions) all passed can change, so only those
are sent to the jury; for the rest the reward is already 0. The jury reward of a run is the
product of its non-NL reward components and the majority verdict (all assertions met -> 1).

Writes runs/text/<run>/jury.json and prints agreement statistics (Fleiss' kappa for the three
judges, Cohen's kappa of each judge against the majority and against the in-run judge).

    python -m eval.jury runs/text/gen2-* runs/text/gen3-* [--dry-run] [--max-usd 5]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGES = (
    "bedrock/qwen.qwen3-next-80b-a3b",
    "bedrock/google.gemma-3-27b-it",
    "bedrock/openai.gpt-oss-120b-1:0",
)

SYSTEM = """TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent (a hospital front-desk assistant) and a caller.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually, judging only what the agent said and did.

FORMAT
Reply with ONLY a JSON object, no prose and no code fences:
{"results": [{"expectedOutcome": "<the expected outcome, copied>", "reasoning": "<short>", "metExpectation": true or false}]}
Return exactly one result per expected outcome, in the same order as given."""


def transcript(sim: dict) -> str:
    return "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in sim.get("messages") or [])


def parse_verdicts(text: str, n: int) -> list[bool] | None:
    """Parse a judge reply into n booleans, or None if it can't be read."""
    if not text:
        return None
    t = re.sub(r"<(reasoning|think|thinking|analysis)>.*?</\1>", "", text, flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or len(results) != n:
        return None
    out = []
    for r in results:
        v = r.get("metExpectation") if isinstance(r, dict) else None
        if isinstance(v, str):
            v = {"true": True, "false": False}.get(v.strip().lower())
        if not isinstance(v, bool):
            return None
        out.append(v)
    return out


class Meter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.usd = 0.0
        self.calls = 0

    def add(self, model: str, usage) -> None:
        from scripts.run_text import price_for

        p = price_for(model)
        pin = getattr(usage, "prompt_tokens", 0) or 0
        pout = getattr(usage, "completion_tokens", 0) or 0
        with self.lock:
            self.usd += p.cost(pin, pout)
            self.calls += 1


def ask(model: str, sim: dict, assertions: list[str], meter: Meter) -> list[bool] | None:
    import litellm

    user = f"conversation:\n{transcript(sim)}\n\nexpectedOutcomes:\n{json.dumps(assertions, ensure_ascii=False)}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(2):
        try:
            resp = litellm.completion(model=model, messages=messages, temperature=0, max_tokens=1500)
        except Exception as e:  # noqa: BLE001 -- a failed call is retried once, then recorded
            if attempt:
                print(f"  judge {model} failed: {str(e)[:120]}", file=sys.stderr)
            continue
        meter.add(model, resp.usage)
        verdicts = parse_verdicts(resp.choices[0].message.content or "", len(assertions))
        if verdicts is not None:
            return verdicts
        messages = messages + [
            {"role": "assistant", "content": resp.choices[0].message.content or ""},
            {"role": "user", "content": f"Reply with only the JSON object, exactly {len(assertions)} results."},
        ]
    return None


def candidates(run_dir: Path) -> list[tuple[dict, list[str], float, list[bool]]]:
    """(sim, assertions, non-NL reward, in-run verdicts) for runs the jury can change."""
    sims = json.loads((run_dir / "results.json").read_text(encoding="utf-8")).get("simulations", [])
    out = []
    for s in sims:
        ri = s.get("reward_info") or {}
        nl = ri.get("nl_assertions") or []
        if not nl or ri.get("reward") is None:
            continue
        other = 1.0
        for k, v in (ri.get("reward_breakdown") or {}).items():
            if k != "NL_ASSERTION":
                other *= float(v)
        if other != 1.0:
            continue
        out.append((s, [n["nl_assertion"] for n in nl], other, [bool(n["met"]) for n in nl]))
    return out


def cohen(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss(rows: list[list[bool]]) -> float:
    """Fleiss' kappa for binary labels; rows = one list of rater labels per item."""
    if not rows:
        return float("nan")
    k = len(rows[0])
    n = len(rows)
    p_yes = sum(sum(r) for r in rows) / (n * k)
    pe = p_yes**2 + (1 - p_yes) ** 2
    pbar = sum((sum(r) * (sum(r) - 1) + (k - sum(r)) * (k - sum(r) - 1)) / (k * (k - 1)) for r in rows) / n
    return 1.0 if pe == 1 else (pbar - pe) / (1 - pe)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="count what would be judged; call nothing")
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop starting judge calls past this")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import litellm

    litellm.modify_params = True
    litellm.suppress_debug_info = True
    os.environ.setdefault("LITELLM_LOG", "ERROR")

    work = []
    for d in args.runs:
        if (d / "results.json").exists():
            for c in candidates(d):
                work.append((d, *c))
    print(f"{len(work)} runs to re-judge x {len(JUDGES)} judges")
    if args.dry_run:
        return 0

    meter = Meter()
    results: dict[Path, dict] = {}
    lock = threading.Lock()

    def job(item):
        d, sim, assertions, other, inrun = item
        if meter.usd > args.max_usd:
            return
        verdicts = {j: ask(j, sim, assertions, meter) for j in JUDGES}
        ok = [v for v in verdicts.values() if v is not None]
        majority = None
        if len(ok) >= 2:  # need at least two readable judges
            majority = [sum(v[i] for v in ok) * 2 > len(ok) for i in range(len(assertions))]
        key = f"{sim['task_id']}#{int(sim.get('trial') or 0)}"
        with lock:
            results.setdefault(d, {})[key] = {
                "assertions": assertions,
                "in_run": inrun,
                "judges": {j.removeprefix("bedrock/"): v for j, v in verdicts.items()},
                "majority": majority,
                "jury_reward": None if majority is None else other * float(all(majority)),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, work))

    # Write per-run files and agreement stats.
    rows, by_judge, inrun_all, maj_all = [], {j: [] for j in JUDGES}, [], []
    changed = 0
    for d, res in results.items():
        (d / "jury.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for r in res.values():
            if r["majority"] is None:
                continue
            labels = [r["judges"][j.removeprefix("bedrock/")] for j in JUDGES]
            if any(v is None for v in labels):
                continue
            for i in range(len(r["assertions"])):
                row = [v[i] for v in labels]
                rows.append(row)
                for j, v in zip(JUDGES, row):
                    by_judge[j].append(v)
                inrun_all.append(r["in_run"][i])
                maj_all.append(r["majority"][i])
            changed += float(all(r["in_run"])) != r["jury_reward"]
    print(f"judge cost ${meter.usd:.2f} over {meter.calls} calls")
    print(f"assertions with all 3 verdicts: {len(rows)}; unanimous: {sum(len(set(r)) == 1 for r in rows)}")
    print(f"Fleiss kappa (3 judges): {fleiss(rows):.3f}")
    for j in JUDGES:
        print(f"  Cohen kappa {j.removeprefix('bedrock/')} vs majority: {cohen(by_judge[j], maj_all):.3f}")
    print(f"  Cohen kappa in-run judge vs majority: {cohen(inrun_all, maj_all):.3f}")
    print(f"runs whose reward changes under the jury: {changed}")
    summary = {
        "judges": [j.removeprefix("bedrock/") for j in JUDGES],
        "assertions_rated": len(rows),
        "unanimous": sum(len(set(r)) == 1 for r in rows),
        "fleiss_kappa": fleiss(rows),
        "cohen_vs_majority": {j.removeprefix("bedrock/"): cohen(by_judge[j], maj_all) for j in JUDGES},
        "cohen_in_run_vs_majority": cohen(inrun_all, maj_all),
        "runs_changed": changed,
        "cost_usd": meter.usd,
    }
    (ROOT / "runs" / "text" / "jury-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
