"""
Complete validation report frame: all Berichte rows per patient merged with predictions.

Skipped / prefilter-negative reports are model decisions (klasse=0) and must remain in the
manual validation cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.berichte_filters import (
    REPORT_TYPES_FOR_MATRIX,
    is_dokumentationsblatt,
    normalize_bertyp,
)
from src.preprocessing.berichte_mapper import read_berichte_csv_robust
from src.preprocessing.evidence_extraction import METHOD_NO_EVIDENCE

LOGGER = logging.getLogger(__name__)

MERGE_KEYS = ("PatientenID", "bericht", "bertyp")

PREDICTION_FILL_DEFAULTS: Dict[str, object] = {
    "klasse": 0,
    "signalstaerke": "niedrig",
    "delir_probability_estimate": 0,
    "manual_review_candidate": "False",
    "decision_rule_applied": "",
    "evidence_snippets": "[]",
    "delir_signale": "",
    "kontext": "",
    "begruendung": "",
    "original_report_text_length": 0,
    "llm_report_text_length": 0,
    "llm_text_reduction_method": "",
    "llm_skipped_by_prefilter": False,
}


def _filter_included_berichte(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_patient_id_column(df.copy())
    if "bertyp" not in out.columns:
        out["bertyp"] = ""
    out["bertyp"] = out["bertyp"].map(normalize_bertyp)
    out["bericht"] = out["bericht"].astype(str).str.strip()
    out = out[~out["bertyp"].map(is_dokumentationsblatt)].copy()
    out = out[out["bertyp"].isin(REPORT_TYPES_FOR_MATRIX)].copy()
    return out


def _filter_included_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return _filter_included_berichte(predictions)


def load_included_berichte_reports(
    berichte_path: Path,
    *,
    patient_ids: Optional[Sequence[str]] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per included Berichte report (optional filter to *patient_ids*)."""
    if berichte_df is not None:
        raw = normalize_patient_id_column(berichte_df.copy())
    else:
        if not berichte_path.exists():
            return pd.DataFrame(columns=list(MERGE_KEYS) + ["berdat"])
        raw = normalize_patient_id_column(
            read_berichte_csv_robust(berichte_path, log_context="validation cohort spine")
        )
    raw.columns = [str(c).strip() for c in raw.columns]
    if "PatientID" in raw.columns and "PatientenID" not in raw.columns:
        raw = raw.rename(columns={"PatientID": "PatientenID"})
    if "bername" in raw.columns and "bericht" not in raw.columns:
        raw["bericht"] = raw["bername"].astype(str).str.strip()
    elif "bericht" not in raw.columns:
        raw["bericht"] = ""
    out = _filter_included_berichte(raw)
    if patient_ids is not None:
        pset = {str(p) for p in patient_ids}
        out = out[out["PatientenID"].isin(pset)].copy()
    if "berdat" not in out.columns:
        out["berdat"] = ""
    return out.drop_duplicates(list(MERGE_KEYS), keep="first")


def derive_report_processing_fields(row: pd.Series) -> Dict[str, object]:
    """Map prediction row to ``status``, ``llm_called``, ``skipped_reason``."""
    in_predictions = bool(row.get("_has_prediction_row"))
    llm_skipped = str(row.get("llm_skipped_by_prefilter", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    method = str(row.get("llm_text_reduction_method") or "").strip()
    rule = str(row.get("decision_rule_applied") or "").strip()
    kontext = str(row.get("kontext") or "")

    if not in_predictions:
        return {
            "status": "missing_prediction",
            "llm_called": 0,
            "skipped_reason": "not_in_prediction_export",
        }

    if llm_skipped or method == METHOD_NO_EVIDENCE or rule == "no_evidence_prefilter_skip":
        return {
            "status": "skipped",
            "llm_called": 0,
            "skipped_reason": rule or METHOD_NO_EVIDENCE,
        }

    if kontext.startswith("Pipeline-Fehler:"):
        return {
            "status": "failed",
            "llm_called": 1,
            "skipped_reason": "pipeline_error",
        }

    return {
        "status": "processed",
        "llm_called": 1,
        "skipped_reason": rule,
    }


def apply_processing_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_has_prediction_row" not in out.columns:
        out["_has_prediction_row"] = True
    derived = out.apply(derive_report_processing_fields, axis=1, result_type="expand")
    out["status"] = derived["status"]
    out["llm_called"] = derived["llm_called"].astype(int)
    out["skipped_reason"] = derived["skipped_reason"]
    return out


def build_complete_validation_reports_frame(
    predictions: pd.DataFrame,
    selected_patient_ids: Sequence[str],
    *,
    berichte_path: Optional[Path] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    All included Berichte reports for selected patients, left-joined with predictions.

    Returns (merged frame, stats dict).
    """
    pids = [str(p) for p in selected_patient_ids]
    preds = _filter_included_predictions(predictions)
    preds = preds[preds["PatientenID"].isin(pids)].copy()
    preds["_has_prediction_row"] = True

    spine_available = berichte_df is not None or (
        berichte_path is not None and berichte_path.exists()
    )
    berichte = (
        load_included_berichte_reports(
            berichte_path or Path("."),
            patient_ids=pids,
            berichte_df=berichte_df,
        )
        if spine_available
        else pd.DataFrame()
    )

    if not spine_available and not preds.empty:
        LOGGER.warning(
            "Berichte.csv not available; cohort uses prediction export only (%d rows). "
            "Re-export after full run_pipeline on complete Berichte for unbiased validation.",
            len(preds),
        )
        merged = preds.copy()
        merged["_has_prediction_row"] = True
        merged = apply_processing_fields(merged.drop(columns=["_has_prediction_row"], errors="ignore"))
        if "klasse" in merged.columns:
            merged["model_report_prediction"] = (
                pd.to_numeric(merged["klasse"], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
        return merged, {
            "berichte_reports": 0,
            "prediction_reports": len(preds),
            "merged_reports": len(merged),
            "only_in_berichte": 0,
            "only_in_predictions": 0,
        }

    if berichte.empty and not preds.empty:
        LOGGER.warning(
            "Berichte spine empty; validation cohort uses prediction rows only (%d).",
            len(preds),
        )
        merged = preds.copy()
        stats = {
            "berichte_reports": 0,
            "prediction_reports": len(preds),
            "merged_reports": len(merged),
            "only_in_berichte": 0,
            "only_in_predictions": len(preds),
        }
        return apply_processing_fields(merged.drop(columns=["_has_prediction_row"], errors="ignore")), stats

    if berichte.empty:
        return pd.DataFrame(), {
            "berichte_reports": 0,
            "prediction_reports": 0,
            "merged_reports": 0,
            "only_in_berichte": 0,
            "only_in_predictions": 0,
        }

    merged = berichte.merge(
        preds,
        on=list(MERGE_KEYS),
        how="left",
        suffixes=("", "_pred"),
    )
    if "_has_prediction_row" not in merged.columns:
        merged["_has_prediction_row"] = False
    merged["_has_prediction_row"] = merged["_has_prediction_row"].fillna(False)

    for col, default in PREDICTION_FILL_DEFAULTS.items():
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default)

    only_berichte = int((~merged["_has_prediction_row"]).sum())
    only_preds = 0
    if not preds.empty and not berichte.empty:
        m = preds[list(MERGE_KEYS)].drop_duplicates().merge(
            berichte[list(MERGE_KEYS)].drop_duplicates(),
            on=list(MERGE_KEYS),
            how="left",
            indicator=True,
        )
        only_preds = int((m["_merge"] == "left_only").sum())

    if only_berichte:
        LOGGER.info(
            "Validation cohort: %d report(s) in Berichte but not in predictions export "
            "(e.g. not processed in run_pipeline / MAX_REPORTS cap).",
            only_berichte,
        )

    stats = {
        "berichte_reports": len(berichte),
        "prediction_reports": len(preds),
        "merged_reports": len(merged),
        "only_in_berichte": only_berichte,
        "only_in_predictions": only_preds,
    }
    merged = apply_processing_fields(merged)
    if "klasse" in merged.columns:
        merged["model_report_prediction"] = (
            pd.to_numeric(merged["klasse"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )
    merged = merged.drop(columns=["_has_prediction_row"], errors="ignore")
    drop_suffix = [c for c in merged.columns if c.endswith("_pred")]
    merged = merged.drop(columns=drop_suffix, errors="ignore")
    return merged, stats


def cohort_processing_summary_lines(cohort: pd.DataFrame) -> List[str]:
    """Summary lines for cohort export report."""
    if cohort.empty:
        return ["Processing summary: (empty cohort)"]
    lines = [
        "",
        "Processing summary (all evaluatable reports per patient)",
        "-" * 44,
        f"total_report_rows={len(cohort)}",
    ]
    if "status" in cohort.columns:
        for status, cnt in cohort["status"].value_counts().sort_index().items():
            lines.append(f"  status={status}: {cnt}")
    if "llm_called" in cohort.columns:
        lines.append(f"  llm_called=1: {int((cohort['llm_called'] == 1).sum())}")
        lines.append(f"  llm_called=0: {int((cohort['llm_called'] == 0).sum())}")
    if "skipped_reason" in cohort.columns:
        top = cohort.loc[cohort["status"] == "skipped", "skipped_reason"].value_counts().head(8)
        if not top.empty:
            lines.append("  skipped_reason (top):")
            for reason, cnt in top.items():
                r = str(reason).strip() or "(empty)"
                lines.append(f"    {r}: {cnt}")
        pred_col = (
            "model_report_prediction"
            if "model_report_prediction" in cohort.columns
            else "klasse"
        )
        guard = cohort.loc[
            (cohort["status"] == "processed")
            & (pd.to_numeric(cohort[pred_col], errors="coerce").fillna(0) == 0),
            "skipped_reason",
        ].value_counts().head(5)
        if not guard.empty:
            lines.append("  guardrail / processed-negative rules (top):")
            for reason, cnt in guard.items():
                r = str(reason).strip() or "(empty)"
                lines.append(f"    {r}: {cnt}")
    return lines
