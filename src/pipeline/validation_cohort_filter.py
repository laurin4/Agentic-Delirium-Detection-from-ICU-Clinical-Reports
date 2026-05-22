"""
Filter pipeline reports to the frozen manual validation cohort (VALIDATION_COHORT_ONLY mode).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.pipeline.paths import FROZEN_PATIENT_VALIDATION_COHORT_PATH
from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.berichte_filters import normalize_bertyp
from src.preprocessing.report_identity import (
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
)

LOGGER = logging.getLogger(__name__)

FALLBACK_KEY_COLUMNS = ("PatientenID", "bertyp", "berdat", "bericht")


def validation_cohort_only_enabled() -> bool:
    """True when ``VALIDATION_COHORT_ONLY=true`` (case-insensitive)."""
    return os.environ.get("VALIDATION_COHORT_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


@dataclass
class CohortFilterSpec:
    """Lookup sets built from ``patient_validation_cohort_frozen.csv``."""

    source_row_ids: Set[str] = field(default_factory=set)
    fallback_keys: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    validation_report_ids: Set[str] = field(default_factory=set)
    cohort_row_count: int = 0
    filter_mode: str = "none"

    def matches_report(self, report: dict) -> bool:
        sid = _norm(report.get(SOURCE_REPORT_ROW_ID_COL))
        if sid and sid in self.source_row_ids:
            return True
        pid = _norm(report.get("PatientenID"))
        bertyp = normalize_bertyp(report.get("bertyp", ""))
        berdat = _norm(report.get("berdat"))
        bericht = _norm(report.get("bericht"))
        key = (pid, bertyp, berdat, bericht)
        return key in self.fallback_keys


def load_frozen_validation_cohort(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
) -> pd.DataFrame:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Frozen validation cohort missing: {cohort_path}. "
            "Run export_patient_validation_cohort and freeze_validation_cohort first."
        )
    df = normalize_patient_id_column(pd.read_csv(cohort_path))
    if df.empty:
        raise ValueError(f"Frozen validation cohort is empty: {cohort_path}")
    return df


def build_cohort_filter_spec(cohort_df: pd.DataFrame) -> CohortFilterSpec:
    """
    Build filter from frozen cohort.

    Primary: ``source_report_row_id``. Fallback: (PatientenID, bertyp, berdat, bericht).
    Also accepts ``pipeline_bericht`` as the bericht component when present.
    """
    spec = CohortFilterSpec(cohort_row_count=len(cohort_df))
    has_source = SOURCE_REPORT_ROW_ID_COL in cohort_df.columns

    for _, row in cohort_df.iterrows():
        vid = _norm(row.get("validation_report_id"))
        if vid:
            spec.validation_report_ids.add(vid)

        sid = _norm(row.get(SOURCE_REPORT_ROW_ID_COL)) if has_source else ""
        if sid:
            spec.source_row_ids.add(sid)

        pid = _norm(row.get("PatientenID"))
        bertyp = normalize_bertyp(row.get("bertyp", ""))
        berdat = _norm(row.get("berdat"))
        bericht_vals = {_norm(row.get("bericht"))}
        if PIPELINE_BERICHT_COL in row.index:
            pber = _norm(row.get(PIPELINE_BERICHT_COL))
            if pber:
                bericht_vals.add(pber)
        for b in bericht_vals:
            if b:
                spec.fallback_keys.add((pid, bertyp, berdat, b))

    if spec.source_row_ids:
        spec.filter_mode = "source_report_row_id"
    elif spec.fallback_keys:
        spec.filter_mode = "patientenid_bertyp_berdat_bericht"
    else:
        spec.filter_mode = "validation_report_id_only"

    return spec


def filter_report_records_for_validation_cohort(
    report_records: Sequence[dict],
    *,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    cohort_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[dict], CohortFilterSpec]:
    """Return only pipeline records that belong to the frozen validation cohort."""
    cohort = cohort_df if cohort_df is not None else load_frozen_validation_cohort(cohort_path)
    spec = build_cohort_filter_spec(cohort)
    filtered = [r for r in report_records if spec.matches_report(r)]
    LOGGER.info(
        "VALIDATION_COHORT_ONLY: %d / %d reports selected (frozen rows=%d, mode=%s)",
        len(filtered),
        len(report_records),
        spec.cohort_row_count,
        spec.filter_mode,
    )
    if len(filtered) < spec.cohort_row_count:
        LOGGER.warning(
            "VALIDATION_COHORT_ONLY: fewer pipeline records (%d) than frozen cohort rows (%d). "
            "Some Berichte rows may lack processable text blocks.",
            len(filtered),
            spec.cohort_row_count,
        )
    return filtered, spec
