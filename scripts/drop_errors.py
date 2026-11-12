"""Remove infrastructure-error simulations from run folders so `--resume` re-runs them.

tau2's resume skips every (task, trial) already in results.json, including runs that failed
for infrastructure reasons (e.g. the AWS SSO token expired mid-run). This drops those entries
(keeping a backup, results.json.bak-<n>) so the next `run_text.py --resume` retries them.
Graded runs (pass or fail) are never touched.

    python scripts/drop_errors.py runs/text/<run> [...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def drop(run_dir: Path) -> int:
    p = run_dir / "results.json"
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    sims = data.get("simulations", [])
    keep = [s for s in sims if s.get("termination_reason") != "infrastructure_error"]
    removed = len(sims) - len(keep)
    if removed:
        n = 1
        while (run_dir / f"results.json.bak-{n}").exists():
            n += 1
        shutil.copy2(p, run_dir / f"results.json.bak-{n}")
        data["simulations"] = keep
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")  # ASCII: tau2 reads with the locale codec on Windows
    return removed


def main(argv: list[str]) -> int:
    total = 0
    for a in argv:
        n = drop(Path(a))
        if n:
            print(f"{a}: removed {n} infrastructure-error run(s)")
        total += n
    print(f"removed {total} in total")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
