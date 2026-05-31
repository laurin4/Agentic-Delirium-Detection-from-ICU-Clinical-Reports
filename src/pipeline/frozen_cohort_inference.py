"""
Build pipeline input records directly from the frozen validation cohort.

Cohort-only inference does NOT filter full Berichte.csv by source_report_row_id.
Report text is taken from the cohort row or resolved once via stable
(PatientenID, bertyp, berdat, bericht) keys from raw Berichte.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import BERICHTE_INPUT_PATH, FROZEN_PATIENT_VALIDATION_COHORT_PATH
from src.pipeline.schema_normalize import normalize_patient_id_column
from src.pipeline.validation_cohort_filter import load_frozen_validation_cohort
from src.pipeline.validation_report_identity import (
    VALIDATION_PATIENT_ID_COL,
    VALIDATION_REPORT_ID_COL,
    assert_validation_report_id_unique,
)
from src.preprocessing.berichte_filters import normalize_bertyp
from src.preprocessing.berichte_mapper import _row_blocks, load_berichte_dataframe
from src.preprocessing.report_identity import (
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
    compute_pipeline_bericht_id,
)

LOGGER = logging.getLogger(__name__)

StableReportKey = Tuple[str, str, str, str]


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def build_stable_report_text_index(
    berichte_path: Path = BERICHTE_INPUT_PATH,
) -> Dict[StableReportKey, str]:
    """
    Map (PatientenID, bertyp, berdat, bericht) → stitched report text.

    Does not use source_report_row_id (positional and unstable across CSV versions).
    """
    if not berichte_path.exists():
        LOGGER.warning("Berichte.csv missing; report text index empty: %s", berichte_path)
        return {}

    df = normalize_patient_id_column(load_berichte_dataframe(berichte_path))
    index: Dict[StableReportKey, str] = {}
    for _, row in df.iterrows():
        row_dict = {c: row.get(c, "") for c in df.columns}
        text = _row_blocks(row_dict)
        if not str(text or "").strip():
            continue
        pid = _norm(row.get("PatientenID"))
        bertyp = normalize_bertyp(row.get("bertyp", ""))
        berdat = _norm(row.get("berdat"))
        bericht_candidates = {
            _norm(row.get("bername")),
            _norm(row.get("bericht")),
            compute_pipeline_bericht_id(row),
        }
        for bericht in bericht_candidates:
            if pid and bericht:
                key = (pid, bertyp, berdat, bericht)
                index.setdefault(key, text)
    return index


def resolve_frozen_cohort_report_text(
    row: pd.Series,
    text_index: Dict[StableReportKey, str],
) -> str:
    """Report text from cohort storage or stable Berichte lookup."""
    for colocated in ("report_text", "llm_report_text"):
        stored = _norm(row.get(colocated))
        if stored:
            return stored

    pid = _norm(row.get("PatientenID"))
    bertyp = normalize_bertyp(row.get("bertyp", ""))
    berdat = _norm(row.get("berdat"))
    bericht_vals = {_norm(row.get("bericht"))}
    if PIPELINE_BERICHT_COL in row.index:
        bericht_vals.add(_norm(row.get(PIPELINE_BERICHT_COL)))

    for bericht in bericht_vals:
        if not bericht:
            continue
        key = (pid, bertyp, berdat, bericht)
        if key in text_index:
            return text_index[key]
    return ""


def build_pipeline_records_from_frozen_cohort(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    *,
    cohort_df: Optional[pd.DataFrame] = None,
) -> List[dict]:
    """
    One pipeline record per frozen cohort row, keyed by validation_report_id.

    Processes exactly the rows in patient_validation_cohort_frozen.csv.
    """
    cohort = cohort_df if cohort_df is not None else load_frozen_validation_cohort(cohort_path)
    assert_validation_report_id_unique(cohort, context="frozen cohort")

    if VALIDATION_REPORT_ID_COL not in cohort.columns:
        raise ValueError(f"Frozen cohort missing {VALIDATION_REPORT_ID_COL}: {cohort_path}")
    if VALIDATION_PATIENT_ID_COL not in cohort.columns:
        raise ValueError(f"Frozen cohort missing {VALIDATION_PATIENT_ID_COL}: {cohort_path}")

    text_index = build_stable_report_text_index(berichte_path)
    records: List[dict] = []
    missing_text: List[str] = []

    for _, row in cohort.iterrows():
        vid = _norm(row.get(VALIDATION_REPORT_ID_COL))
        vpid = _norm(row.get(VALIDATION_PATIENT_ID_COL))
        if not vid:
            raise ValueError("Frozen cohort row with empty validation_report_id")

        bericht = _norm(row.get("bericht")) or _norm(row.get(PIPELINE_BERICHT_COL))
        report_text = resolve_frozen_cohort_report_text(row, text_index)
        if not report_text.strip():
            missing_text.append(vid)

        records.append(
            {
                VALIDATION_REPORT_ID_COL: vid,
                VALIDATION_PATIENT_ID_COL: vpid,
                "PatientenID": _norm(row.get("PatientenID")),
                "bericht": bericht,
                "bertyp": normalize_bertyp(row.get("bertyp", "")),
                "berdat": _norm(row.get("berdat")),
                SOURCE_REPORT_ROW_ID_COL: _norm(row.get(SOURCE_REPORT_ROW_ID_COL)),
                "report_text": report_text,
            }
        )

    if missing_text:
        LOGGER.warning(
            "Frozen cohort inference: %d / %d rows have empty report_text "
            "(stable Berichte lookup failed): %s",
            len(missing_text),
            len(records),
            missing_text[:5],
        )

    LOGGER.info(
        "Built %d pipeline records from frozen cohort (%s)",
        len(records),
        cohort_path.name,
    )
    return records
