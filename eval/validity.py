"""Validity checks on finished runs: did the language manipulation work, and was the harness fair?

    python -m eval.validity runs/text/gen2-* runs/text/gen3-* [--json out.json]

1. Caller language fidelity. A caller turn counts as Hinglish if it contains at least two common
   Hindi words written in Latin script (or any Devanagari). Hinglish tasks should be almost all
   Hinglish; English tasks should be almost none (otherwise the matched pairs do not isolate
   code-mixing).
2. Agent language mirroring. Share of agent replies to Hinglish callers that themselves contain
   Hindi, per model, and the pass rate when the agent mirrors vs. stays in English.
3. Harness fairness. Premature hang-ups and false transfers removed by the harness, per English and
   Hinglish conversation (from summary.json; counts cover the sessions that recorded them).
4. Call length. Agent and caller turns per conversation, by language and outcome.

The Hindi word list is deliberately small and excludes words that are also common in English or
Indian English ("main", "to", "do", "he", "so", "the", "par", "papa", "beta"), so it under-counts
rather than over-counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HINDI = {
    "hai", "hain", "hoon", "hun", "tha", "thi", "mujhe", "mera", "meri", "mere", "humein",
    "aap", "aapka", "aapki", "aapke", "apna", "apni", "kya", "kyun", "kaise", "kab", "kahan",
    "nahi", "nahin", "chahiye", "karna", "karni", "karo", "kariye", "kijiye", "karein", "kar",
    "ke", "ki", "ka", "ko", "se", "bhi", "toh", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "kal", "abhi", "theek", "thik", "haan", "ji", "accha", "acha", "achha", "bahut", "wala",
    "wali", "wale", "dikhana", "batao", "bataiye", "sakte", "sakta", "sakti", "dijiye", "liye",
    "aur", "lekin", "mein", "yeh", "woh", "kuch", "sab", "ek", "dono", "baat",
    "bolo", "boliye", "samajh", "dhanyavaad", "shukriya", "namaste",
}
WORD = re.compile(r"[A-Za-z]+")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
CONTROL = re.compile(r"###[A-Z]+###")


def is_hinglish(text: str, min_words: int = 2) -> bool:
    if DEVANAGARI.search(text):
        return True
    return sum(w.lower() in HINDI for w in WORD.findall(text)) >= min_words


def turns(sim: dict, role: str) -> list[str]:
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") != role:
            continue
        t = CONTROL.sub("", m.get("content") or "").strip()
        if len(WORD.findall(t)) >= 4:  # skip "yes", "okay", bare tokens
            out.append(t)
    return out


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def analyse(paths: list[Path]) -> dict:
    caller = {"en": [0, 0], "hi": [0, 0]}
    mirror = defaultdict(lambda: {"turns": [0, 0], "calls": defaultdict(lambda: [0, 0])})
    guards = {"en": defaultdict(int), "hi": defaultdict(int)}
    calls = {"en": 0, "hi": 0}
    length = defaultdict(lambda: [0, 0])  # (lang, passed) -> [agent_turns_sum, calls]
    for d in paths:
        rp, sp = d / "results.json", d / "summary.json"
        if not rp.exists():
            continue
        summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        model = (summary.get("agent_llm") or d.name).removeprefix("bedrock/")
        for key in ("premature_stops_blocked_by_run", "false_transfers_blocked_by_run"):
            for run, n in (summary.get(key) or {}).items():
                lang = "hi" if run.split("#")[0].endswith("-hi") else "en"
                guards[lang][key] += n
        for sim in json.loads(rp.read_text(encoding="utf-8")).get("simulations", []):
            lang = "hi" if sim["task_id"].endswith("-hi") else "en"
            calls[lang] += 1
            ct = turns(sim, "user")
            caller[lang][0] += sum(map(is_hinglish, ct))
            caller[lang][1] += len(ct)
            passed = (sim.get("reward_info") or {}).get("reward") == 1.0
            at = turns(sim, "assistant")
            length[(lang, passed)][0] += len([m for m in sim.get("messages") or [] if m.get("role") == "assistant"])
            length[(lang, passed)][1] += 1
            if lang == "hi" and at:
                hits = sum(map(is_hinglish, at))
                mirror[model]["turns"][0] += hits
                mirror[model]["turns"][1] += len(at)
                mirrored = hits / len(at) >= 0.5
                c = mirror[model]["calls"][mirrored]
                c[0] += passed
                c[1] += 1
    return {
        "caller_hinglish_turns": {k: rate(*v) for k, v in caller.items()},
        "agent_mirrors_hinglish": {
            m: {
                "turn_rate": rate(*v["turns"]),
                "pass_when_mirroring": rate(*v["calls"][True]),
                "pass_when_english": rate(*v["calls"][False]),
            }
            for m, v in sorted(mirror.items())
        },
        "harness_removals_per_call": {
            lang: {k: round(n / calls[lang], 3) for k, n in g.items()} | {"calls": calls[lang]}
            for lang, g in guards.items()
        },
        "agent_messages_per_call": {
            f"{lang}_{'pass' if p else 'fail'}": round(s / n, 1) for (lang, p), (s, n) in sorted(length.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = analyse([p for p in a.runs if p.is_dir()])
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
