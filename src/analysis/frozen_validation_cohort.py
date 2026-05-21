"""Frozen manual validation cohort paths, checksums, and resolution for evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

FROZEN_VALIDATION_NOTE = (
    "This frozen cohort is the fixed dataset for manual validation."
)

OVERWRITE_FROZEN_VALIDATION_ENV = "OVERWRITE_FROZEN_VALIDATION"


def overwrite_frozen_validation_enabled() -> bool:
    return os.environ.get(OVERWRITE_FROZEN_VALIDATION_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_validation_dir() -> Path:
    from src.pipeline.paths import FROZEN_VALIDATION_COHORT_DIR

    return FROZEN_VALIDATION_COHORT_DIR


def frozen_cohort_paths() -> Tuple[Path, Path, Path]:
    from src.pipeline.paths import (
        FROZEN_COHORT_METADATA_PATH,
        FROZEN_MANUAL_REPORT_LABELS_PATH,
        FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    )

    return (
        FROZEN_PATIENT_VALIDATION_COHORT_PATH,
        FROZEN_MANUAL_REPORT_LABELS_PATH,
        FROZEN_COHORT_METADATA_PATH,
    )


def frozen_cohort_exists() -> bool:
    cohort_path, _, _ = frozen_cohort_paths()
    return cohort_path.exists()


def assert_can_write_frozen(*, force: bool = False) -> None:
    """Block overwrite unless ``OVERWRITE_FROZEN_VALIDATION=true`` or *force*."""
    if force or not frozen_cohort_exists():
        return
    if overwrite_frozen_validation_enabled():
        LOGGER.warning(
            "Overwriting existing frozen validation cohort (%s=true).",
            OVERWRITE_FROZEN_VALIDATION_ENV,
        )
        return
    cohort_path, labels_path, meta_path = frozen_cohort_paths()
    raise FileExistsError(
        "Frozen validation cohort already exists at "
        f"{cohort_path.parent}. "
        f"Set {OVERWRITE_FROZEN_VALIDATION_ENV}=true to overwrite, "
        "or remove the frozen_validation_cohort/ directory manually."
    )


def build_frozen_metadata(
    cohort_path: Path,
    labels_path: Path,
    *,
    frozen_cohort_out: Path,
    frozen_labels_out: Path,
    predictions_source: Path,
    baseline_source: Path,
) -> Dict[str, Any]:
    cohort_df = pd.read_csv(cohort_path)
    labels_df = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    patient_col = (
        "validation_patient_id"
        if "validation_patient_id" in cohort_df.columns
        else "PatientenID"
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_unique_patients": int(cohort_df[patient_col].nunique()),
        "n_report_rows": int(len(cohort_df)),
        "n_label_rows": int(len(labels_df)),
        "source_prediction_file": str(predictions_source.resolve()),
        "source_baseline_file": str(baseline_source.resolve()),
        "source_cohort_file": str(cohort_path.resolve()),
        "source_labels_file": str(labels_path.resolve()) if labels_path.exists() else None,
        "frozen_cohort_file": str(frozen_cohort_out.resolve()),
        "frozen_labels_file": str(frozen_labels_out.resolve()),
        "checksum_sha256": {
            "patient_validation_cohort": sha256_file(cohort_path),
            "manual_report_labels": sha256_file(labels_path) if labels_path.exists() else None,
            "patient_validation_cohort_frozen": sha256_file(frozen_cohort_out),
            "manual_report_labels_frozen": sha256_file(frozen_labels_out)
            if frozen_labels_out.exists()
            else None,
        },
        "note": FROZEN_VALIDATION_NOTE,
    }


def resolve_validation_input_paths(
    cohort_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], bool]:
    """
    Resolve cohort/labels paths for manual validation evaluation.

    When frozen files exist, they take precedence over mutable exports unless
    explicit paths are passed and frozen cohort is absent.
    """
    from src.pipeline.paths import (
        FROZEN_MANUAL_REPORT_LABELS_PATH,
        FROZEN_PATIENT_VALIDATION_COHORT_PATH,
        MANUAL_REPORT_LABELS_PATH,
        PATIENT_VALIDATION_COHORT_PATH,
    )

    using_frozen = False
    if FROZEN_PATIENT_VALIDATION_COHORT_PATH.exists():
        resolved_cohort = FROZEN_PATIENT_VALIDATION_COHORT_PATH
        using_frozen = True
        if cohort_path is not None and cohort_path != FROZEN_PATIENT_VALIDATION_COHORT_PATH:
            LOGGER.info(
                "Using frozen cohort %s (ignoring explicit cohort_path).",
                FROZEN_PATIENT_VALIDATION_COHORT_PATH,
            )
    else:
        resolved_cohort = cohort_path or PATIENT_VALIDATION_COHORT_PATH

    if labels_path is not None:
        resolved_labels = labels_path
    elif using_frozen and FROZEN_MANUAL_REPORT_LABELS_PATH.exists():
        resolved_labels = FROZEN_MANUAL_REPORT_LABELS_PATH
    elif MANUAL_REPORT_LABELS_PATH.exists():
        resolved_labels = MANUAL_REPORT_LABELS_PATH
    else:
        resolved_labels = (
            FROZEN_MANUAL_REPORT_LABELS_PATH
            if using_frozen
            else MANUAL_REPORT_LABELS_PATH
        )

    if using_frozen:
        LOGGER.info("Manual validation evaluation using frozen validation cohort.")

    return resolved_cohort, resolved_labels, using_frozen
