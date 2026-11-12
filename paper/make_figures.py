"""Draw the paper's results figure from paper/generated/results.json (run make_results first).

    python -m paper.make_figures            # needs matplotlib (pip install matplotlib)

Writes paper/figures/results.pdf: (a) English vs Hinglish pass rate per agent on the 240 text
tasks, (b) text vs speech-channel pass rate per agent on the 80 batch-1 tasks; whiskers are
Wilson 95% intervals. Colours: the first three slots of a colour-blind-validated categorical
palette (blue, orange, aqua); every number is also in the paper's tables.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "generated" / "results.json"
OUT = ROOT / "paper" / "figures" / "results.pdf"

PRETTY = {
    "global.anthropic.claude-sonnet-4-6": "Sonnet\n4.6",
    "deepseek.v3.2": "DeepSeek\nV3.2",
    "global.amazon.nova-2-lite-v1:0": "Nova 2\nLite",
    "mistral.mistral-large-3-675b-instruct": "Mistral\nLarge 3",
    "qwen.qwen3-next-80b-a3b": "Qwen3\nNext 80B",
}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def bars(ax, groups, series, title):
    """groups: list of (label, [rate dicts per series]); series: list of (name, colour)."""
    n = len(series)
    width = 0.8 / n
    gap = 0.02  # a thin surface-coloured gap between adjacent bars
    for i, (name, colour) in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for g, (_, cells) in enumerate(groups):
            c = cells[i]
            if c is None:
                continue
            x = g - 0.4 + width * (i + 0.5)
            xs.append(x)
            ys.append(100 * c["rate"])
            lo.append(100 * (c["rate"] - c["ci95"][0]))
            hi.append(100 * (c["ci95"][1] - c["rate"]))
        ax.bar(xs, ys, width=width - gap, color=colour, label=name, zorder=2, linewidth=0)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=2,
                    capthick=0.8, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=6.5, color=INK, linespacing=1.1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Pass rate (%)", fontsize=8, color=INK)
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=16)
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.99), ncol=2,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0,
              labelcolor=INK)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    models = sorted(r["models"], key=lambda m: -r["models"][m]["overall"]["rate"])
    lang = [(PRETTY.get(m, m), [r["models"][m]["by_language"]["en"], r["models"][m]["by_language"]["hinglish"]])
            for m in models]
    voice = r.get("voice_vs_text") or {}
    vmodels = [m for m in models if m in voice]
    speech = [(PRETTY.get(m, m), [voice[m]["text"], voice[m]["voice"]]) for m in vmodels]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [5, 4]})
    n_tasks = r["models"][models[0]]["overall"]["n"]
    bars(axes[0], lang, [("English", BLUE), ("Hinglish", ORANGE)],
         f"(a) Text, {n_tasks // 2} matched pairs")
    if speech:
        n_v = max(v["n"] for v in voice.values())
        bars(axes[1], speech, [("Text", BLUE), ("Speech channel", AQUA)],
             f"(b) Same {n_v} tasks: text vs. speech")
        axes[1].set_ylabel("")
    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT} (+ .png preview)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
