"""
Copy current V1 baseline validation outputs into prompt_runs/v1/run_01/.

Read-only on source files: copies only, never moves or deletes.
Never touches manual_report_labels_frozen.csv or frozen cohort files.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    MATCHING_AUDIT_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.prompt_run_paths import PROMPT_RUNS_ROOT

LOGGER = logging.getLogger(__name__)

V1_RUN_01_DIR = PROMPT_RUNS_ROOT / "v1" / "run_01"

COPY_TARGETS: tuple[tuple[Path, Path], ...] = (
    (
        VALIDATION_COHORT_PREDICTIONS_PATH,
        V1_RUN_01_DIR / "predictions" / "validation_cohort_predictions.csv",
    ),
    (
        FINAL_MANUAL_VALIDATION_EVAL_DIR,
        V1_RUN_01_DIR / "final_evaluation",
    ),
    (
        MANUAL_VALIDATION_DIR / "final_eval_alignment_check.txt",
        V1_RUN_01_DIR / "audits" / "final_eval_alignment_check.txt",
    ),
    (
        MATCHING_AUDIT_DIR,
        V1_RUN_01_DIR / "audits" / "matching_audit",
    ),
    (
        MANUAL_VALIDATION_DIR / "matching_audit_positive",
        V1_RUN_01_DIR / "audits" / "matching_audit_positive",
    ),
)

PROTECTED_PATHS: frozenset[Path] = frozenset(
    {
        FROZEN_MANUAL_REPORT_LABELS_PATH,
        FROZEN_PATIENT_VALIDATION_COHORT_PATH,
        FROZEN_MANUAL_REPORT_LABELS_PATH.parent / "frozen_cohort_metadata.json",
    }
)


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for protected in PROTECTED_PATHS:
        if resolved == protected.resolve():
            raise ValueError(f"Refusing to copy protected frozen file: {path}")


def copy_path(src: Path, dst: Path) -> bool:
    """Copy file or directory tree. Returns True if copy performed."""
    _assert_not_protected(src)
    if not src.exists():
        LOGGER.warning("Skip missing source: %s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    LOGGER.info("Copied %s -> %s", src, dst)
    return True


def archive_current_v1_results(
    targets: Sequence[tuple[Path, Path]] = COPY_TARGETS,
) -> list[tuple[Path, Path]]:
    """Copy legacy validation outputs to v1/run_01. Returns list of successful copies."""
    copied: list[tuple[Path, Path]] = []
    for src, dst in targets:
        if copy_path(src, dst):
            copied.append((src, dst))
    return copied


def format_archive_report(copied: Iterable[tuple[Path, Path]]) -> str:
    lines = [
        "Archive current V1 baseline results",
        "=" * 44,
        f"destination_root={V1_RUN_01_DIR}",
        "",
    ]
    items = list(copied)
    if not items:
        lines.append("No files copied (sources missing?).")
    else:
        lines.append(f"copied_items={len(items)}")
        for src, dst in items:
            lines.append(f"  {src} -> {dst}")
    lines.extend(
        [
            "",
            "Protected (never copied):",
            f"  {FROZEN_MANUAL_REPORT_LABELS_PATH}",
            f"  {FROZEN_PATIENT_VALIDATION_COHORT_PATH}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    copied = archive_current_v1_results()
    report = format_archive_report(copied)
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
