"""
Pre-commit safety check — block sensitive paths from being committed.

Inspects files currently staged for commit (via `git diff --cached --name-only`)
and exits with code 1 if any of them match risky patterns. No external
dependencies. Designed to be run manually:

    python scripts/check_no_sensitive_files.py

Returns:
    0 if no staged file matches a risky pattern (or nothing is staged)
    1 if at least one staged file is risky
    2 on environment errors (git not available, not a repo, ...)
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from typing import List, Tuple


# Path-prefix risks: any staged path that starts with one of these is blocked.
RISKY_PREFIXES: Tuple[str, ...] = (
    "data/raw/",
    "data/processed/",
    "data/anonymized/",
    "data/structured/raw/",
    "data/structured/processed/",
    "Data/",
    "outputs/",
    "reports/",
    "plots/",
    "models/",
    "models_Qwen/",
    "models_Ollama/",
    "wheelhouse/",
    "wheelhouse_linux/",
    "Ba_venv/",
    "Ba_venv_backup/",
    "delirium_env/",
    ".venv/",
    "venv/",
    "env/",
)

# Glob risks: matched against the file path (case-insensitive).
RISKY_GLOBS: Tuple[str, ...] = (
    "*.csv",
    "*.xlsx",
    "*.xls",
    "*.parquet",
    "*.jsonl",
    "*.tar.gz",
    "*.token",
    "*.key",
    "*.pem",
    ".env",
    "*.env",
    "*.png",
    "*.pdf",
)

# Files explicitly allowed even if they would otherwise match (e.g. .gitkeep).
ALLOWED_EXACT: Tuple[str, ...] = (
    "data/.gitkeep",
    "data/raw/.gitkeep",
    "outputs/.gitkeep",
)


def _staged_files() -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("ERROR: 'git' command not found in PATH.", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        print("ERROR: git command failed:", exc.stderr.strip(), file=sys.stderr)
        sys.exit(2)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _is_risky(path: str) -> Tuple[bool, str]:
    if path in ALLOWED_EXACT:
        return False, ""
    norm = path.replace("\\", "/")
    for prefix in RISKY_PREFIXES:
        if norm.startswith(prefix):
            return True, f"path under '{prefix}'"
    lower = norm.lower()
    for pattern in RISKY_GLOBS:
        if fnmatch.fnmatch(lower, pattern.lower()):
            return True, f"matches '{pattern}'"
    return False, ""


def main() -> int:
    staged = _staged_files()
    if not staged:
        print("OK: no files staged.")
        return 0

    blocked: List[Tuple[str, str]] = []
    for path in staged:
        risky, reason = _is_risky(path)
        if risky:
            blocked.append((path, reason))

    if blocked:
        print("BLOCKED: the following staged files look sensitive and must not be committed:")
        for path, reason in blocked:
            print(f"  - {path}    ({reason})")
        print("")
        print("To unstage:")
        print("  git restore --staged <file>")
        print("Or update .gitignore and run 'git reset' to start clean.")
        return 1

    print(f"OK: {len(staged)} staged file(s); none match risky patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
