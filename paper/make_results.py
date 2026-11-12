"""Build the paper's result tables and numbers from benchmark runs (no hand-copied numbers).

    python -m paper.make_results runs/text/gen2-* runs/text/gen3-*

Writes paper/generated/results.tex (macros + tables) and paper/generated/results.json.
Every number in the paper's Results section comes from here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval import score as sc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
TASK_FILES = [
    ROOT / "domains" / "hospital_in" / "data" / "tasks_generated.json",
    ROOT / "domains" / "hospital_in" / "data" / "tasks.json",
]
WRITES = {
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "register_patient",
    "escalate_emergency",
    "transfer_to_human_agent",
}
SAFETY_WRITES = {"book_appointment", "reschedule_appointment", "cancel_appointment"}

PRETTY = {
    "global.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek.v3.2": "DeepSeek V3.2",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
    "global.amazon.nova-2-lite-v1:0": "Amazon Nova 2 Lite",
    "qwen.qwen3-next-80b-a3b": "Qwen3-Next 80B",
}
CAT_LABEL = {
    "booking": "Book",
    "reschedule": "Resch.",
    "report": "Report",
    "redflag": "Red flag",
    "privacy": "Privacy",
    "routing": "Routing",
    "panic": "Panic",
}


def tasks_by_id() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in TASK_FILES:
        if p.exists():
            for t in json.loads(p.read_text(encoding="utf-8")):
                out[t["id"]] = t
    return out


def note(task: dict, key: str) -> str | None:
    for part in task["description"]["notes"].split(";"):
        k, _, v = part.strip().partition("=")
        if k == key:
            return v
    return None


def failure_mode(task: dict, sim: dict) -> str:
    """One coarse label per failed run, from the agent's write actions and the grader."""
    cat = note(task, "category")
    gold = [a["name"] for a in task["evaluation_criteria"].get("actions") or []]
    did = [
        t["name"]
        for m in sim.get("messages") or []
        for t in (m.get("tool_calls") or [])
        if t["name"] in WRITES
    ]
    ri = sim.get("reward_info") or {}
    nl_fail = any(not n.get("met") for n in (ri.get("nl_assertions") or []))
    if sim.get("termination_reason") == "max_steps":
        return "call did not end (step limit)"
    if "escalate_emergency" in gold:  # red flag, or panic with a red flag
        if "escalate_emergency" not in did:
            return "missed emergency"
        if SAFETY_WRITES & set(did):
            return "emergency + routine change"
        if "transfer_to_human_agent" in did:
            return "emergency + front-desk transfer"
        return "emergency, other"
    if gold == ["transfer_to_human_agent"]:
        if "escalate_emergency" in did:
            return "false emergency"
        return "no transfer"
    if cat == "privacy":
        gold_args = [a["arguments"] for a in task["evaluation_criteria"].get("actions") or []]
        gold_ids = {a.get("uhid") or a.get("appointment_id") for a in gold_args}
        acted = [
            t["arguments"].get("uhid") or t["arguments"].get("appointment_id")
            for m in sim.get("messages") or []
            for t in (m.get("tool_calls") or [])
            if t["name"] in SAFETY_WRITES
        ]
        if any(a not in gold_ids for a in acted):
            return "acted for unverified person"
        if nl_fail:
            return "disclosed protected info"
        return "own request wrong/missing"
    if cat == "report" and nl_fail:
        return "disclosed protected info"
    if "escalate_emergency" in did:
        return "false emergency"
    if not (set(did) & SAFETY_WRITES):
        if "transfer_to_human_agent" in did:
            return "gave up (transferred)"
        return "no change made"
    if "register_patient" in did and "register_patient" not in gold:
        return "duplicate/invented registration"
    return "wrong slot/doctor/department"


def collect(paths: list[Path], jury: bool = False):
    tasks = tasks_by_id()
    runs = [sc.load_run(p, jury) for p in paths if (p / "results.json").exists()]
    merged = sc.merge_by_model(runs)
    result = sc.score(merged)
    modes: dict[str, Counter] = defaultdict(Counter)
    for p in paths:
        rp = p / "results.json"
        if not rp.exists():
            continue
        r = sc.load_run(p)
        for sim in json.loads(rp.read_text(encoding="utf-8")).get("simulations", []):
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward is None or reward == 1.0:
                continue
            t = tasks.get(sim["task_id"])
            if t is not None:
                modes[r.model][failure_mode(t, sim)] += 1
    result["failure_modes"] = {m: dict(c) for m, c in modes.items()}
    return result


def tex_pct(r: dict, ci: bool = False) -> str:
    if not r or not r["n"]:
        return "--"
    s = f"{100 * r['rate']:.0f}"
    if ci:
        lo, hi = r["ci95"]
        s += f" {{\\scriptsize[{100 * lo:.0f}--{100 * hi:.0f}]}}"
    return s


def macro_name(s: str) -> str:
    return "".join(ch for ch in s.title() if ch.isalpha())


def render_tex(result: dict) -> str:
    models = sorted(result["models"].items(), key=lambda kv: -kv[1]["overall"]["rate"])
    L: list[str] = ["% Generated by paper/make_results.py -- do not edit by hand."]
    n_tasks = max(m["overall"]["n"] for _, m in models)
    total_cost = sum(m["cost_usd"] or 0 for _, m in models)
    L += [
        f"\\newcommand{{\\NTasks}}{{{n_tasks}}}",
        f"\\newcommand{{\\NModels}}{{{len(models)}}}",
        f"\\newcommand{{\\NRuns}}{{{sum(m['overall']['n'] for _, m in models)}}}",
        f"\\newcommand{{\\TotalCost}}{{\\${total_cost:.0f}}}",
    ]
    for model, m in models:
        k = macro_name(PRETTY.get(model, model))
        L.append(f"\\newcommand{{\\Overall{k}}}{{{tex_pct(m['overall'])}\\%}}")
        L.append(f"\\newcommand{{\\Privacy{k}}}{{{tex_pct(m['by_category'].get('privacy'))}\\%}}")
        L.append(f"\\newcommand{{\\Redflag{k}}}{{{tex_pct(m['by_category'].get('redflag'))}\\%}}")
    pp = result["pooled_pairs"]
    L += [
        f"\\newcommand{{\\PairsEnOnly}}{{{pp['en_pass_hi_fail']}}}",
        f"\\newcommand{{\\PairsHiOnly}}{{{pp['en_fail_hi_pass']}}}",
        f"\\newcommand{{\\PairsP}}{{{pp['mcnemar_p']:.3f}}}",
        "",
    ]
    # Main table
    cats = list(CAT_LABEL)
    L += [
        "\\newcommand{\\MainTable}{%",
        "\\begin{table*}[t]\\centering\\small",
        "\\caption{Pass rate (\\%) per agent on the \\NTasks{} generated tasks (one trial each). "
        "Overall shows a Wilson 95\\% interval. Categories pool English and Hinglish twins. "
        "Cost is the metered Bedrock cost of all calls in the run (agent, caller and judge).}",
        "\\label{tab:main}",
        "\\begin{tabular}{l c " + "c" * len(cats) + " c c r}",
        "\\toprule",
        "Agent & Overall & " + " & ".join(CAT_LABEL[c] for c in cats) + " & EN & Hinglish & Cost \\\\",
        "\\midrule",
    ]
    for model, m in models:
        cells = [tex_pct(m["by_category"].get(c)) for c in cats]
        lg = [tex_pct(m["by_language"].get(x)) for x in ("en", "hinglish")]
        cost = f"\\${m['cost_usd']:.2f}" if m["cost_usd"] is not None else "--"
        L.append(
            f"{PRETTY.get(model, model)} & {tex_pct(m['overall'], ci=True)} & "
            + " & ".join(cells + lg)
            + f" & {cost} \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}}", ""]
    # Pairs table
    L += [
        "\\newcommand{\\PairsTable}{%",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Matched English/Hinglish pairs that disagree. $b$: English passed, "
        "Hinglish failed; $c$: the reverse. $p$: exact two-sided McNemar test.}",
        "\\label{tab:pairs}",
        "\\begin{tabular}{l r r r}",
        "\\toprule",
        "Agent & $b$ & $c$ & $p$ \\\\",
        "\\midrule",
    ]
    for model, m in models:
        p = m["hinglish_pairs"]
        L.append(
            f"{PRETTY.get(model, model)} & {p['en_pass_hi_fail']} & {p['en_fail_hi_pass']} & "
            f"{p['mcnemar_p']:.2f} \\\\"
        )
    L += [
        "\\midrule",
        f"Pooled & {pp['en_pass_hi_fail']} & {pp['en_fail_hi_pass']} & {pp['mcnemar_p']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}}",
        "",
    ]
    # Failure modes table
    modes = result.get("failure_modes", {})
    all_modes = sorted({k for c in modes.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in modes.values()))
    L += [
        "\\newcommand{\\FailureTable}{%",
        "\\begin{table*}[t]\\centering\\small",
        "\\caption{Failed runs by failure mode (counts), labelled automatically from the agent's "
        "write actions and the grader's checks.}",
        "\\label{tab:failures}",
        "\\begin{tabular}{l " + "r" * len(models) + "}",
        "\\toprule",
        "Failure mode & " + " & ".join(PRETTY.get(mo, mo) for mo, _ in models) + " \\\\",
        "\\midrule",
    ]
    for k in all_modes:
        L.append(
            k.replace("&", "\\&")
            + " & "
            + " & ".join(str(modes.get(mo, {}).get(k, 0)) for mo, _ in models)
            + " \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}}", ""]
    return "\n".join(L)


def paired_conditions(text_paths: list[Path], voice_paths: list[Path], jury: bool) -> dict:
    """Same agent, same tasks: text vs speech channel. Discordant pairs + exact McNemar."""
    text = {r.model: r for r in sc.merge_by_model([sc.load_run(p, jury) for p in text_paths if (p / "results.json").exists()])}
    voice = {r.model: r for r in sc.merge_by_model([sc.load_run(p, jury) for p in voice_paths if (p / "results.json").exists()])}
    out = {}
    for model, v in voice.items():
        t = text.get(model)
        if t is None:
            continue
        keys = sorted(t.outcome.keys() & v.outcome.keys())
        b = sum(t.outcome[k] and not v.outcome[k] for k in keys)
        c = sum(v.outcome[k] and not t.outcome[k] for k in keys)
        rf = [k for k in keys if k[0].startswith("redflag")]
        hi = [k for k in keys if k[0].endswith("-hi")]
        out[model] = {
            "n": len(keys),
            "text": sc.rate([t.outcome[k] for k in keys]),
            "voice": sc.rate([v.outcome[k] for k in keys]),
            "text_only": b,
            "voice_only": c,
            "mcnemar_p": sc.mcnemar_exact(b, c),
            "redflag_text": sc.rate([t.outcome[k] for k in rf]),
            "redflag_voice": sc.rate([v.outcome[k] for k in rf]),
            "hinglish_text": sc.rate([t.outcome[k] for k in hi]),
            "hinglish_voice": sc.rate([v.outcome[k] for k in hi]),
        }
    # A run cut short (e.g. by a provider's daily token cap) is not comparable to the complete
    # ones: report it as pending rather than on a smaller task set.
    full = max((v["n"] for v in out.values()), default=0)
    for model in [m for m, v in out.items() if v["n"] < full]:
        print(f"voice: {model} has {out[model]['n']}/{full} paired tasks; left out as incomplete", file=sys.stderr)
        del out[model]
    return out


def passk_table(paths: list[Path], jury: bool) -> dict:
    runs = sc.merge_by_model([sc.load_run(p, jury) for p in paths if (p / "results.json").exists()])
    res = sc.score(runs)
    out = {
        m: {"pass1": v["overall"], "pass_k": v["pass_hat_k"],
            "redflag_pass1": v["by_category"].get("redflag"), "privacy_pass1": v["by_category"].get("privacy")}
        for m, v in res["models"].items()
    }
    # As for speech: a run cut short (daily token cap) is reported as pending, not on fewer trials.
    full = max((v["pass1"]["n"] for v in out.values()), default=0)
    for model in [m for m, v in out.items() if v["pass1"]["n"] < full or not v["pass_k"]]:
        print(f"pass^k: {model} has {out[model]['pass1']['n']}/{full} trials; left out as incomplete",
              file=sys.stderr)
        del out[model]
    return out


def render_extra_tex(voice: dict, passk: dict, hearing: dict | None) -> str:
    # Empty fallbacks so main.tex always compiles while runs are still going.
    L = [
        f"\\newcommand{{\\{name}}}{{\\todo{{{name} pending}}}}"
        for name, have in (("VoiceTable", voice), ("PassKTable", passk), ("HearingTable", hearing))
        if not have
    ]
    for m, v in voice.items():
        k = macro_name(PRETTY.get(m, m))
        L += [
            f"\\newcommand{{\\VoiceText{k}}}{{{v['text']['passed']}}}",
            f"\\newcommand{{\\VoiceSpeech{k}}}{{{v['voice']['passed']}}}",
            f"\\newcommand{{\\VoiceHiText{k}}}{{{v['hinglish_text']['passed']}}}",
            f"\\newcommand{{\\VoiceHiSpeech{k}}}{{{v['hinglish_voice']['passed']}}}",
        ]
    if voice:
        L.append(f"\\newcommand{{\\VoiceN}}{{{max(v['n'] for v in voice.values())}}}")
    for m, v in passk.items():
        k = macro_name(PRETTY.get(m, m))
        if v["pass_k"]:
            L.append(f"\\newcommand{{\\PassFour{k}}}{{{tex_pct(v['pass_k'])}\\%}}")
            L.append(f"\\newcommand{{\\PassOne{k}}}{{{tex_pct(v['pass1'])}\\%}}")
    if hearing:
        def hv(key, field):  # noqa: E306
            c = hearing.get(key, {}).get(field)
            return f"{100 * c[0]:.0f}\\%" if c else "--"
        L += [
            f"\\newcommand{{\\HearEmergEnIn}}{{{hv('phone_noisy|en-IN|hinglish', 'redflag_recognisable')}}}",
            f"\\newcommand{{\\HearEmergAuto}}{{{hv('phone_noisy|auto|hinglish', 'redflag_recognisable')}}}",
            f"\\newcommand{{\\HearDobEnIn}}{{{hv('phone_noisy|en-IN|hinglish', 'dob')}}}",
            f"\\newcommand{{\\HearDobAuto}}{{{hv('phone_noisy|auto|hinglish', 'dob')}}}",
        ]
    if voice:
        L += [
            "\\newcommand{\\VoiceTable}{%",
            "\\begin{table}[t]\\centering\\small",
            "\\caption{Same agent, same 80 tasks: text vs.\\ hearing the caller through Polly, a "
            "noisy 8\\,kHz phone channel (15\\,dB SNR) and Transcribe. $b$/$c$: tasks passed only in "
            "text / only with speech; $p$: exact McNemar.}",
            "\\label{tab:voice}",
            "\\begin{tabular}{l r r r r r}",
            "\\toprule",
            "Agent & Text & Speech & $b$ & $c$ & $p$ \\\\",
            "\\midrule",
        ]
        for m, v in sorted(voice.items(), key=lambda kv: -kv[1]["text"]["rate"]):
            L.append(
                f"{PRETTY.get(m, m)} & {tex_pct(v['text'])} & {tex_pct(v['voice'])} & "
                f"{v['text_only']} & {v['voice_only']} & {v['mcnemar_p']:.2f} \\\\"
            )
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}}", ""]
    if passk:
        L += [
            "\\newcommand{\\PassKTable}{%",
            "\\begin{table}[t]\\centering\\small",
            "\\caption{Reliability on the 80 batch-1 tasks, 4 trials each: pass$^1$ (all trials) "
            "and pass$^4$ (share of tasks passed in all four trials), with Wilson 95\\% intervals.}",
            "\\label{tab:passk}",
            "\\begin{tabular}{l c c}",
            "\\toprule",
            "Agent & pass$^1$ & pass$^4$ \\\\",
            "\\midrule",
        ]
        for m, v in sorted(passk.items(), key=lambda kv: -kv[1]["pass1"]["rate"]):
            pk = v["pass_k"]
            L.append(f"{PRETTY.get(m, m)} & {tex_pct(v['pass1'], ci=True)} & "
                     f"{tex_pct(pk, ci=True) if pk else '--'} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}}", ""]
    if hearing:
        L += [
            "\\newcommand{\\HearingTable}{%",
            "\\begin{table*}[t]\\centering\\small",
            "\\caption{Hearing test: caller lines spoken by Polly, passed through the phone channel "
            "and transcribed by Transcribe with three language settings. Identity: share of clips "
            "whose UHID / mobile digits survive exactly, and whose name / date of birth an LLM "
            "checker can recover (script-independent). Emergency: share of red-flag descriptions "
            "still recognisable as the same emergency.}",
            "\\label{tab:hearing}",
            "\\begin{tabular}{l l l r r r r r}",
            "\\toprule",
            "Channel & ASR setting & Caller & UHID & Mobile & Name & DOB & Emergency \\\\",
            "\\midrule",
        ]
        for key, cells in hearing.items():
            cond, cfg, lang = key.split("|")
            def g(k):  # noqa: E306
                v = cells.get(k)
                return f"{100 * v[0]:.0f}" if v else "--"
            L.append(
                f"{cond.replace('_', ' ')} & {cfg} & {'Hinglish' if lang == 'hinglish' else 'English'} & "
                f"{g('uhid')} & {g('phone')} & {g('name')} & {g('dob')} & {g('redflag_recognisable')} \\\\"
            )
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}}", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="text runs on the generated tasks (pass^1)")
    ap.add_argument("--voice", nargs="*", type=Path, default=[], help="speech-channel runs (batch 1)")
    ap.add_argument("--paired-text", nargs="*", type=Path, default=[], help="text runs on batch 1")
    ap.add_argument("--passk", nargs="*", type=Path, default=[], help="multi-trial runs")
    ap.add_argument("--hearing", type=Path, help="runs/hearing/score.json")
    ap.add_argument("--jury", action="store_true", help="use 3-judge majority verdicts")
    args = ap.parse_args(argv)
    result = collect(args.runs, jury=args.jury)
    if not result["models"]:
        print("no runs found", file=sys.stderr)
        return 2
    result["voice_vs_text"] = paired_conditions(args.paired_text, args.voice, args.jury) if args.voice else {}
    result["pass_k"] = passk_table(args.passk, args.jury) if args.passk else {}
    hearing = json.loads(args.hearing.read_text(encoding="utf-8")) if args.hearing else None
    result["hearing"] = hearing
    jury_path = ROOT / "runs" / "text" / "jury-summary.json"
    jury_tex = ""
    if jury_path.exists():
        j = json.loads(jury_path.read_text(encoding="utf-8"))
        result["jury"] = j
        jury_tex = "\n".join([
            f"\\newcommand{{\\JuryRated}}{{{j['assertions_rated']}}}",
            f"\\newcommand{{\\JuryUnanimous}}{{{j['unanimous']}}}",
            f"\\newcommand{{\\JuryFleiss}}{{{j['fleiss_kappa']:.2f}}}",
            f"\\newcommand{{\\JuryInRunKappa}}{{{j['cohen_in_run_vs_majority']:.2f}}}",
            f"\\newcommand{{\\JuryChanged}}{{{j['runs_changed']}}}",
        ])
    OUT.mkdir(parents=True, exist_ok=True)
    # README table (Markdown) from the same numbers.
    rows = sorted(result["models"].items(), key=lambda kv: -kv[1]["overall"]["rate"])
    md = ["| Agent | Overall (95% CI) | Red flags | Privacy | Routing | English | Hinglish |",
          "|---|---|---|---|---|---|---|"]
    for m, v in rows:
        f = lambda r: f"{100 * r['rate']:.0f}%" if r and r["n"] else "–"  # noqa: E731
        lo, hi = v["overall"]["ci95"]
        md.append(
            f"| {PRETTY.get(m, m)} | {100 * v['overall']['rate']:.0f}% ({100 * lo:.0f}–{100 * hi:.0f}) | "
            f"{f(v['by_category'].get('redflag'))} | {f(v['by_category'].get('privacy'))} | "
            f"{f(v['by_category'].get('routing'))} | {f(v['by_language'].get('en'))} | "
            f"{f(v['by_language'].get('hinglish'))} |"
        )
    pp = result["pooled_pairs"]
    md.append("")
    md.append(
        f"Matched English/Hinglish pairs (pooled): {pp['en_pass_hi_fail']} passed only in English, "
        f"{pp['en_fail_hi_pass']} only in Hinglish (exact McNemar p = {pp['mcnemar_p']:.3f})."
    )
    (OUT / "readme_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tex = (render_tex(result) + "\n" + render_extra_tex(result["voice_vs_text"], result["pass_k"], hearing)
           + "\n" + jury_tex + "\n")
    # Never wider than the text block: shrink a table only if it would overflow (adjustbox).
    tex = tex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}")
    (OUT / "results.tex").write_text(tex, encoding="utf-8")
    print(sc.markdown(result))
    print("\nfailure modes:", json.dumps(result["failure_modes"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
