"""
Slim manual annotation labels and merge into the full validation cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

SLIM_LABEL_EXPORT_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "validation_report_id",
    "report_nr_within_patient",
    "PatientenID",
    "bertyp",
    "berdat",
    "model_report_prediction",
    "status",
    "llm_called",
    "skipped_reason",
    "evidence_snippets",
    "manual_report_ground_truth",
    "manual_comment",
)

MANUAL_LABEL_MERGE_COLUMNS: tuple[str, ...] = (
    "manual_report_ground_truth",
    "manual_comment",
)


def _normalize_manual_report_gt(value: object) -> Optional[str]:
    """Return ``'0'`` or ``'1'`` when *value* is a valid binary label, else ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s in ("0", "1"):
        return s
    try:
        n = int(float(s))
    except (TypeError, ValueError):
        return None
    if n in (0, 1):
        return str(n)
    return None


def build_manual_report_labels_sheet(cohort: pd.DataFrame) -> pd.DataFrame:
    """Subset of cohort rows with empty manual columns for external annotation."""
    export_cols = [c for c in SLIM_LABEL_EXPORT_COLUMNS if c in cohort.columns]
    missing_required = [
        c
        for c in (
            "validation_patient_id",
            "validation_report_id",
            "PatientenID",
            "model_report_prediction",
        )
        if c not in cohort.columns
    ]
    if missing_required:
        raise ValueError(f"Cohort missing required columns for label export: {missing_required}")
    if cohort["validation_report_id"].duplicated().any():
        raise ValueError("validation_report_id must be unique in patient_validation_cohort.csv")

    out = cohort[export_cols].copy()
    out["manual_report_ground_truth"] = ""
    out["manual_comment"] = ""
    sort_cols = [c for c in ("validation_patient_id", "berdat", "bertyp", "validation_report_id") if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def merge_manual_report_labels(
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    log_context: str = "manual validation evaluation",
) -> pd.DataFrame:
    """
    Merge annotated slim labels into the full cohort by ``validation_report_id``.

    Label file values override cohort manual columns when ``manual_report_ground_truth``
    is ``0`` or ``1``; non-empty ``manual_comment`` in labels overrides the cohort comment.
    """
    if "validation_report_id" not in cohort.columns:
        raise ValueError("Cohort must contain validation_report_id")
    if "validation_report_id" not in labels.columns:
        raise ValueError("Labels must contain validation_report_id")

    dup = labels["validation_report_id"].duplicated()
    if dup.any():
        n = int(dup.sum())
        raise ValueError(
            f"manual_report_labels.csv has {n} duplicate validation_report_id value(s); IDs must be unique."
        )

    lab = labels.drop_duplicates("validation_report_id", keep="first").set_index(
        "validation_report_id"
    )
    cohort_ids = set(cohort["validation_report_id"].astype(str))
    label_ids = set(lab.index.astype(str))
    unknown = label_ids - cohort_ids
    if unknown:
        sample = sorted(unknown)[:5]
        LOGGER.warning(
            "[%s] %d validation_report_id(s) in labels not found in cohort (e.g. %s); ignored.",
            log_context,
            len(unknown),
            sample,
        )

    out = cohort.copy()
    for col in MANUAL_LABEL_MERGE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(object)

    rid = out["validation_report_id"].astype(str)
    if "manual_report_ground_truth" in lab.columns:
        gt_raw = rid.map(lab["manual_report_ground_truth"])
        gt_norm = gt_raw.map(_normalize_manual_report_gt)
        mask = gt_norm.notna()
        out.loc[mask, "manual_report_ground_truth"] = gt_norm.loc[mask].values

    if "manual_comment" in lab.columns:
        comment = rid.map(lab["manual_comment"])
        cmask = comment.notna() & (comment.astype(str).str.strip() != "")
        out.loc[cmask, "manual_comment"] = comment.loc[cmask].astype(str).values

    n_gt = int(out["manual_report_ground_truth"].map(_normalize_manual_report_gt).notna().sum())
    LOGGER.info(
        "[%s] merged manual_report_labels: %d report rows with manual_report_ground_truth 0/1",
        log_context,
        n_gt,
    )
    return out


def load_cohort_for_manual_evaluation(
    cohort_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    *,
    auto_merge_default_labels: bool = True,
    prefer_frozen: bool = True,
) -> pd.DataFrame:
    """
    Load cohort for evaluation; merge label file when present.

    When *prefer_frozen* is true, uses frozen cohort/labels if they exist (see
    ``resolve_validation_input_paths``).
    """
    if prefer_frozen or cohort_path is None:
        from src.analysis.frozen_validation_cohort import resolve_validation_input_paths

        cohort_path, labels_path, _ = resolve_validation_input_paths(
            cohort_path, labels_path
        )
    else:
        from src.pipeline.paths import (
            MANUAL_REPORT_LABELS_PATH,
            PATIENT_VALIDATION_COHORT_PATH,
        )

        cohort_path = cohort_path or PATIENT_VALIDATION_COHORT_PATH
        if labels_path is None and auto_merge_default_labels:
            labels_path = MANUAL_REPORT_LABELS_PATH

    if not cohort_path.exists():
        raise FileNotFoundError(f"Patient validation cohort missing: {cohort_path}")

    cohort = pd.read_csv(cohort_path)
    resolved = labels_path
    if resolved is not None and resolved.exists():
        labels = pd.read_csv(resolved)
        cohort = merge_manual_report_labels(cohort, labels)
    elif labels_path is not None and resolved is not None and not resolved.exists():
        LOGGER.warning("Manual report labels file not found; using cohort as-is: %s", resolved)

    return cohort
