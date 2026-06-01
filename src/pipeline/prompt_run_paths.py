"""
Versioned output paths for prompt-based validation runs (V1/V2, run_01..03).

Environment (both required for versioned outputs):
  DELIRIUM_PROMPT_VERSION=v1|v2
  VALIDATION_RUN_ID=run_01|run_02|run_03

When unset, legacy paths under outputs/predictions/ and manual_validation/ are used.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    MANUAL_VALIDATION_DIR,
    MATCHING_AUDIT_DIR,
    PREDICTIONS_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)

DELIRIUM_PROMPT_VERSION_ENV = "DELIRIUM_PROMPT_VERSION"
VALIDATION_RUN_ID_ENV = "VALIDATION_RUN_ID"

ALLOWED_PROMPT_VERSIONS = frozenset({"v1", "v2"})
_RUN_ID_PATTERN = re.compile(r"^run_\d{2}$")

PROMPT_RUNS_ROOT = MANUAL_VALIDATION_DIR / "prompt_runs"
PROMPT_RUNS_COMPARISON_DIR = PROMPT_RUNS_ROOT / "comparison"


def normalize_prompt_version(version: str) -> str:
    v = version.strip().lower()
    if not v.startswith("v"):
        v = f"v{v}"
    if v not in ALLOWED_PROMPT_VERSIONS:
        allowed = ", ".join(sorted(ALLOWED_PROMPT_VERSIONS))
        raise ValueError(
            f"Invalid {DELIRIUM_PROMPT_VERSION_ENV}='{version}'. Allowed: {allowed}"
        )
    return v


def normalize_run_id(run_id: str) -> str:
    r = run_id.strip().lower().replace("-", "_")
    if re.fullmatch(r"\d{2}", r):
        r = f"run_{r}"
    if not _RUN_ID_PATTERN.match(r):
        raise ValueError(
            f"Invalid {VALIDATION_RUN_ID_ENV}='{run_id}'. "
            "Expected format run_01, run_02, run_03, ..."
        )
    return r


def get_prompt_version_from_env() -> str:
    """Active prompt version; defaults to v1 when env unset."""
    raw = os.environ.get(DELIRIUM_PROMPT_VERSION_ENV, "v1").strip()
    if not raw:
        return "v1"
    return normalize_prompt_version(raw)


def get_validation_run_id_from_env() -> str | None:
    raw = os.environ.get(VALIDATION_RUN_ID_ENV, "").strip()
    if not raw:
        return None
    return normalize_run_id(raw)


def is_versioned_validation_run() -> bool:
    """True when both DELIRIUM_PROMPT_VERSION and VALIDATION_RUN_ID are set."""
    version_raw = os.environ.get(DELIRIUM_PROMPT_VERSION_ENV, "").strip()
    run_raw = os.environ.get(VALIDATION_RUN_ID_ENV, "").strip()
    return bool(version_raw) and bool(run_raw)


def get_prompt_run_dir(
    version: str | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Root directory for one versioned validation run.

    Requires explicit *version* and *run_id*, or both env vars when versioned run is active.
    """
    if version is None:
        version = get_prompt_version_from_env()
    else:
        version = normalize_prompt_version(version)

    if run_id is None:
        run_id = get_validation_run_id_from_env()
        if not run_id:
            raise ValueError(
                f"{VALIDATION_RUN_ID_ENV} is required for a versioned prompt run directory."
            )
    else:
        run_id = normalize_run_id(run_id)

    return PROMPT_RUNS_ROOT / version / run_id


def get_versioned_predictions_path() -> Path:
    """Versioned cohort predictions path; requires both env vars."""
    return (
        get_prompt_run_dir()
        / "predictions"
        / "validation_cohort_predictions.csv"
    )


def resolve_cohort_predictions_output_path() -> Path:
    """Write target for VALIDATION_COHORT_ONLY pipeline (versioned or legacy)."""
    if is_versioned_validation_run():
        out = get_versioned_predictions_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    return VALIDATION_COHORT_PREDICTIONS_PATH


def resolve_validation_predictions_path() -> Path:
    """Read/write target for validation cohort predictions (versioned or legacy)."""
    if is_versioned_validation_run():
        return get_versioned_predictions_path()
    return VALIDATION_COHORT_PREDICTIONS_PATH


def get_versioned_final_eval_dir() -> Path:
    if is_versioned_validation_run():
        return get_prompt_run_dir() / "final_evaluation"
    return FINAL_MANUAL_VALIDATION_EVAL_DIR


def get_versioned_audit_dir() -> Path:
    if is_versioned_validation_run():
        return get_prompt_run_dir() / "audits"
    return MATCHING_AUDIT_DIR


def get_versioned_matching_audit_dir() -> Path:
    base = get_versioned_audit_dir()
    if is_versioned_validation_run():
        return base / "matching_audit"
    return MATCHING_AUDIT_DIR


def get_versioned_positive_matching_audit_dir() -> Path:
    base = get_versioned_audit_dir()
    if is_versioned_validation_run():
        return base / "matching_audit_positive"
    return MANUAL_VALIDATION_DIR / "matching_audit_positive"


def get_versioned_final_eval_alignment_path() -> Path:
    if is_versioned_validation_run():
        return get_versioned_audit_dir() / "final_eval_alignment_check.txt"
    return MANUAL_VALIDATION_DIR / "final_eval_alignment_check.txt"
