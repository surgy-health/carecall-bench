"""Build the public release as a fresh repository with a single commit (nothing is pushed).

    python scripts/export_public.py ../carecall-bench-public

Takes the committed tree at HEAD (so untracked files, `.env`, `runs/` and `docs/plans/` can never
leak), leaves out internal-only files (EXCLUDE), checks the result for secrets and internal
references, and commits it as one "v0.1" commit in a new repository at the target path. Pushing it
to GitHub is a separate, manual step.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Internal working notes for Claude Code sessions (plan-doc links, budget, people); not public.
EXCLUDE = {"CLAUDE.md"}
FORBIDDEN = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"aws_secret_access_key[ \t]*=[ \t]*\S{16,}", re.I),  # a value, not the empty placeholder
    re.compile(r"claude\.ai/artifact/"),
    re.compile(r"\b\d{12}\b(?=.*(arn|account|bucket))", re.I),
]


def git(*args: str, cwd: Path = ROOT) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    target = Path(argv[0]).resolve()
    if target.exists() and any(target.iterdir()):
        print(f"{target} exists and is not empty; choose a new path", file=sys.stderr)
        return 2
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("warning: uncommitted changes are NOT included (export uses HEAD)", file=sys.stderr)

    target.mkdir(parents=True, exist_ok=True)
    archive = tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", "HEAD")))
    problems, kept = [], 0
    for member in archive.getmembers():
        if member.name in EXCLUDE or not member.isfile():
            continue
        data = archive.extractfile(member).read()
        if member.name != "uv.lock":
            text = data.decode("utf-8", errors="ignore")
            problems += [f"{member.name}: {p.pattern}" for p in FORBIDDEN if p.search(text)]
        out = target / member.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        kept += 1
    if problems:
        print("refusing to commit; internal or secret-looking content found:", *problems, sep="\n  ",
              file=sys.stderr)
        return 1

    name = git("config", "user.name").decode().strip()
    email = git("config", "user.email").decode().strip()
    git("init", "-q", "-b", "main", cwd=target)
    git("add", "-A", cwd=target)
    git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m",
        "CareCall-Bench v0.1", cwd=target)
    print(f"public snapshot: {kept} files, 1 commit at {target} (excluded: {', '.join(sorted(EXCLUDE))})")
    print("review it, then create the GitHub repo and push from there.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
