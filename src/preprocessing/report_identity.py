"""
Deterministic report keys shared by run_pipeline, validation cohort export, and merges.

``source_report_row_id`` is assigned on the full loaded Berichte.csv (before bertyp filter).
``pipeline_bericht`` matches the ``bericht`` field written by ``build_report_level_berichte_records``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from src.preprocessing.berichte_filters import normalize_bertyp

SOURCE_REPORT_ROW_ID_COL = "source_report_row_id"
PIPELINE_BERICHT_COL = "pipeline_bericht"
FALLBACK_MERGE_KEYS: tuple[str, ...] = ("PatientenID", "bertyp", "berdat", "bericht")
PRIMARY_MERGE_KEYS: tuple[str, ...] = (SOURCE_REPORT_ROW_ID_COL,)
PIPELINE_MERGE_KEYS: tuple[str, ...] = ("PatientenID", PIPELINE_BERICHT_COL)


def _normalize_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def assign_source_report_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ``berichte_row_<index>`` from row order in *df* (call before report-type filtering)."""
    out = df.reset_index(drop=True)
    out[SOURCE_REPORT_ROW_ID_COL] = "berichte_row_" + out.index.astype(str)
    return out


def compute_pipeline_bericht_id(row: pd.Series) -> str:
    """
    Same rule as ``build_report_level_berichte_records``:

    - ``bername`` when non-empty
    - else ``{bertyp}_{PatientenID}_{pandas_index}``
    """
    bername = _normalize_str(row.get("bername", ""))
    if bername:
        return bername
    pid = _normalize_str(row.get("PatientenID") or row.get("PatientID", ""))
    bertyp = normalize_bertyp(row.get("bertyp", ""))
    idx = row.name
    if idx is None or (isinstance(idx, float) and pd.isna(idx)):
        idx = 0
    return f"{bertyp or 'bericht'}_{pid}_{idx}"


def attach_report_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pipeline_bericht``; normalize ``PatientenID``, ``bertyp``, ``berdat``, ``bericht``."""
    out = df.copy()
    if "PatientID" in out.columns and "PatientenID" not in out.columns:
        out = out.rename(columns={"PatientID": "PatientenID"})
    if "PatientenID" in out.columns:
        out["PatientenID"] = out["PatientenID"].astype(str).str.strip()
    if "bertyp" in out.columns:
        out["bertyp"] = out["bertyp"].map(normalize_bertyp)
    if "bericht" not in out.columns:
        out["bericht"] = ""
    out["bericht"] = out["bericht"].astype(str).str.strip()
    if "berdat" not in out.columns:
        out["berdat"] = ""
    out["berdat"] = out["berdat"].astype(str).str.strip()
    out[PIPELINE_BERICHT_COL] = out.apply(compute_pipeline_bericht_id, axis=1)
    return out


def row_has_report_text_blocks(row: pd.Series) -> bool:
    """True when diag/epikrise/jetziges_leiden/prozedere yield non-empty stitched text."""
    for col in ("diag", "epikrise", "jetziges_leiden", "prozedere"):
        if _normalize_str(row.get(col, "")):
            return True
    return False


def choose_prediction_merge_keys(
    spine: pd.DataFrame, preds: pd.DataFrame
) -> tuple[list[str], str]:
    """
    Return (merge_columns, strategy_name).

    Prefer ``source_report_row_id`` when present in both frames with coverage.
    Else ``PatientenID`` + ``pipeline_bericht`` (legacy predictions use ``bericht`` = pipeline id).
    """
    if (
        SOURCE_REPORT_ROW_ID_COL in spine.columns
        and SOURCE_REPORT_ROW_ID_COL in preds.columns
        and preds[SOURCE_REPORT_ROW_ID_COL].notna().any()
    ):
        covered = preds[SOURCE_REPORT_ROW_ID_COL].isin(spine[SOURCE_REPORT_ROW_ID_COL]).sum()
        if covered > 0:
            return [SOURCE_REPORT_ROW_ID_COL], "source_report_row_id"

    if PIPELINE_BERICHT_COL in spine.columns:
        spine_key = PIPELINE_BERICHT_COL
    else:
        spine_key = "bericht"

    if spine_key in spine.columns and "bericht" in preds.columns:
        return ["PatientenID", spine_key], "patientenid_pipeline_bericht"

    keys = [k for k in FALLBACK_MERGE_KEYS if k in spine.columns and k in preds.columns]
    return keys, "fallback_patientenid_bertyp_berdat_bericht"
