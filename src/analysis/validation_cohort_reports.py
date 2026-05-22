"""
Complete validation report frame: all Berichte rows per patient merged with predictions.

Raw Berichte.csv (after report-type filter) is authoritative. Predictions are LEFT-merged
using ``source_report_row_id`` or pipeline ``bericht`` identity; pipeline ``status`` fields
are preserved when a prediction row matches.
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
from src.preprocessing.report_identity import (
    FALLBACK_MERGE_KEYS,
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
    assign_source_report_row_ids,
    attach_report_identity_columns,
    choose_prediction_merge_keys,
    row_has_report_text_blocks,
)

LOGGER = logging.getLogger(__name__)

# Legacy alias
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
    "status": "",
    "llm_called": 0,
    "skipped_reason": "",
}


def _filter_included_berichte(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_patient_id_column(df.copy())
    if out.empty:
        return out
    if "bertyp" not in out.columns:
        out["bertyp"] = ""
    if "bericht" not in out.columns:
        out["bericht"] = ""
    out["bertyp"] = out["bertyp"].map(normalize_bertyp)
    out["bericht"] = out["bericht"].astype(str).str.strip()
    out = out[~out["bertyp"].map(is_dokumentationsblatt)].copy()
    out = out[out["bertyp"].isin(REPORT_TYPES_FOR_MATRIX)].copy()
    return out


def _filter_included_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return _filter_included_berichte(predictions)


def load_raw_included_report_spine(
    berichte_path: Path,
    *,
    patient_ids: Optional[Sequence[str]] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    All included raw Berichte rows (no deduplication).

    Assigns ``source_report_row_id`` before bertyp filter; adds ``pipeline_bericht`` for merge.
    """
    if berichte_df is not None:
        raw = normalize_patient_id_column(berichte_df.copy())
    else:
        if not berichte_path.exists():
            return pd.DataFrame(
                columns=list(FALLBACK_MERGE_KEYS) + [SOURCE_REPORT_ROW_ID_COL, PIPELINE_BERICHT_COL]
            )
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

    raw = assign_source_report_row_ids(raw)
    out = _filter_included_berichte(raw)
    out = attach_report_identity_columns(out)
    if patient_ids is not None:
        pset = {str(p) for p in patient_ids}
        out = out[out["PatientenID"].isin(pset)].copy()
    return out.reset_index(drop=True)


def load_included_berichte_reports(
    berichte_path: Path,
    *,
    patient_ids: Optional[Sequence[str]] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Backward-compatible alias: full raw spine (no deduplication)."""
    return load_raw_included_report_spine(
        berichte_path, patient_ids=patient_ids, berichte_df=berichte_df
    )


def _prepare_predictions_for_merge(
    preds: pd.DataFrame, spine: pd.DataFrame, merge_on: Sequence[str]
) -> Tuple[pd.DataFrame, str]:
    """
    Normalize predictions and enrich legacy exports with ``source_report_row_id`` when possible.
    """
    if preds.empty:
        return preds, "empty"

    out = normalize_patient_id_column(preds.copy())
    if "bertyp" in out.columns:
        out["bertyp"] = out["bertyp"].map(normalize_bertyp)
    if "bericht" not in out.columns:
        out["bericht"] = ""
    out["bericht"] = out["bericht"].astype(str).str.strip()
    if "berdat" not in out.columns:
        out["berdat"] = ""
    out["berdat"] = out["berdat"].astype(str).str.strip()
    # Pipeline CSV stores the pipeline identifier in ``bericht`` (bername or synthetic id).
    out[PIPELINE_BERICHT_COL] = out["bericht"]

    has_source = (
        SOURCE_REPORT_ROW_ID_COL in out.columns
        and out[SOURCE_REPORT_ROW_ID_COL].astype(str).str.strip().ne("").any()
    )
    if not has_source and SOURCE_REPORT_ROW_ID_COL in spine.columns:
        lookup = spine[
            [SOURCE_REPORT_ROW_ID_COL, "PatientenID", PIPELINE_BERICHT_COL]
        ].drop_duplicates(subset=["PatientenID", PIPELINE_BERICHT_COL], keep="first")
        out = out.drop(columns=[SOURCE_REPORT_ROW_ID_COL], errors="ignore")
        out = out.merge(lookup, on=["PatientenID", PIPELINE_BERICHT_COL], how="left")
        LOGGER.info(
            "Enriched legacy predictions with source_report_row_id from Berichte spine "
            "(PatientenID + pipeline_bericht)."
        )

    cols = [c for c in merge_on if c in out.columns]
    if cols:
        out = out.drop_duplicates(subset=cols, keep="first")
    out["_has_prediction_row"] = True
    return out, "ready"


def derive_report_processing_fields(row: pd.Series) -> Dict[str, object]:
    """Use pipeline CSV status when matched; implicit negative only when unmatched."""
    if bool(row.get("_has_prediction_row")):
        status = str(row.get("status") or "").strip()
        if status in ("skipped", "processed", "failed"):
            return {
                "status": status,
                "llm_called": int(pd.to_numeric(row.get("llm_called"), errors="coerce") or 0),
                "skipped_reason": str(row.get("skipped_reason") or ""),
            }
        # Legacy predictions without status column: infer from pipeline flags
        llm_skipped = str(row.get("llm_skipped_by_prefilter", "")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        method = str(row.get("llm_text_reduction_method") or "").strip()
        rule = str(row.get("decision_rule_applied") or "").strip()
        kontext = str(row.get("kontext") or "")
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

    return {
        "status": "missing_prediction",
        "llm_called": 0,
        "skipped_reason": "missing_prediction_implicit_negative",
    }


def apply_processing_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_has_prediction_row" not in out.columns:
        out["_has_prediction_row"] = False
    derived = out.apply(derive_report_processing_fields, axis=1, result_type="expand")
    out["status"] = derived["status"]
    out["llm_called"] = derived["llm_called"].astype(int)
    out["skipped_reason"] = derived["skipped_reason"]
    return out


def assert_spine_row_count_preserved(
    spine: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    context: str = "validation cohort export",
) -> None:
    expected = len(spine)
    actual = len(merged)
    if expected == actual:
        return

    spine_counts = spine.groupby("PatientenID", dropna=False).size()
    merged_counts = merged.groupby("PatientenID", dropna=False).size()
    all_pids = sorted(set(spine_counts.index.astype(str)) | set(merged_counts.index.astype(str)))
    lines = [
        f"{context}: raw spine rows={expected}, exported rows={actual} (must be equal).",
        "Per-patient raw vs exported row counts:",
    ]
    for pid in all_pids:
        raw_n = int(spine_counts.get(pid, 0))
        exp_n = int(merged_counts.get(pid, 0))
        flag = "OK" if raw_n == exp_n else "MISMATCH"
        lines.append(f"  PatientenID={pid}: raw={raw_n} exported={exp_n} [{flag}]")
    raise ValueError("\n".join(lines))


def diagnose_missing_prediction_reasons(
    spine: pd.DataFrame,
    preds: pd.DataFrame,
    merged: pd.DataFrame,
) -> Dict[str, int]:
    """Heuristic counts for why spine rows lack a prediction match."""
    unmatched = merged[~merged["_has_prediction_row"].fillna(False).astype(bool)].copy()
    if unmatched.empty:
        return {}

    reasons: Dict[str, int] = {}
    for _, row in unmatched.iterrows():
        if not row_has_report_text_blocks(row):
            reasons["no_text_blocks_in_raw_row"] = reasons.get("no_text_blocks_in_raw_row", 0) + 1
            continue
        pid = str(row.get("PatientenID", ""))
        pber = str(row.get(PIPELINE_BERICHT_COL, row.get("bericht", "")))
        if preds.empty:
            reasons["predictions_empty"] = reasons.get("predictions_empty", 0) + 1
            continue
        pred_sub = preds[preds["PatientenID"].astype(str) == pid]
        if pred_sub.empty:
            reasons["patient_not_in_predictions"] = reasons.get("patient_not_in_predictions", 0) + 1
        elif "bericht" in pred_sub.columns and not (
            pred_sub["bericht"].astype(str).str.strip() == pber
        ).any():
            reasons["pipeline_bericht_mismatch"] = reasons.get("pipeline_bericht_mismatch", 0) + 1
        else:
            reasons["other_unmatched"] = reasons.get("other_unmatched", 0) + 1
    return reasons


def _merge_predictions_onto_spine(
    spine: pd.DataFrame, preds: pd.DataFrame
) -> Tuple[pd.DataFrame, str, str]:
    spine_n = len(spine)
    if preds.empty:
        merged = spine.copy()
        merged["_has_prediction_row"] = False
        assert_spine_row_count_preserved(spine, merged)
        return merged, "none", "predictions_empty"

    merge_on, strategy = choose_prediction_merge_keys(spine, preds)
    if strategy != "source_report_row_id":
        LOGGER.warning(
            "Predictions lack usable source_report_row_id; merging via %s (%s). "
            "Re-run run_pipeline to add source_report_row_id for best traceability.",
            strategy,
            ", ".join(merge_on),
        )

    preds_ready, _ = _prepare_predictions_for_merge(preds, spine, merge_on)

    if strategy == "patientenid_pipeline_bericht":
        if PIPELINE_BERICHT_COL not in preds_ready.columns and "bericht" in preds_ready.columns:
            preds_ready[PIPELINE_BERICHT_COL] = preds_ready["bericht"].astype(str).str.strip()
        merge_on = ["PatientenID", PIPELINE_BERICHT_COL]

    spine_cols = set(spine.columns)
    pred_extra = [c for c in preds_ready.columns if c not in spine_cols or c in merge_on]
    if strategy == "patientenid_pipeline_bericht" and PIPELINE_BERICHT_COL in pred_extra:
        pred_extra = [c for c in pred_extra if c != "bericht"]

    merged = spine.merge(
        preds_ready[pred_extra],
        on=list(merge_on),
        how="left",
        validate="m:1",
        suffixes=("", "_pred"),
    )
    merged["_has_prediction_row"] = merged["_has_prediction_row"].fillna(False).astype(bool)

    assert_spine_row_count_preserved(spine, merged)
    if len(merged) != spine_n:
        raise ValueError(
            f"Prediction merge changed row count: spine={spine_n} merged={len(merged)} "
            f"(merge_on={merge_on}, strategy={strategy})"
        )
    return merged, strategy, ", ".join(merge_on)


def per_patient_spine_export_counts(
    spine: pd.DataFrame, cohort: pd.DataFrame
) -> pd.DataFrame:
    raw = spine.groupby("PatientenID", dropna=False).size().rename("raw_included_reports")
    if cohort.empty:
        exp = pd.Series(dtype=int, name="exported_reports")
    else:
        exp = cohort.groupby("PatientenID", dropna=False).size().rename("exported_reports")
    out = pd.concat([raw, exp], axis=1).fillna(0).astype(int)
    out["match"] = out["raw_included_reports"] == out["exported_reports"]
    return out.sort_index()


def build_complete_validation_reports_frame(
    predictions: pd.DataFrame,
    selected_patient_ids: Sequence[str],
    *,
    berichte_path: Optional[Path] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    All included Berichte reports for selected patients, left-joined with predictions.

    Returns (merged frame, stats dict). Row count equals raw spine for selected patients.
    """
    pids = [str(p) for p in selected_patient_ids]
    preds = _filter_included_predictions(predictions)
    if not preds.empty:
        preds = preds[preds["PatientenID"].isin(pids)].copy()

    spine_available = berichte_df is not None or (
        berichte_path is not None and berichte_path.exists()
    )
    berichte = (
        load_raw_included_report_spine(
            berichte_path or Path("."),
            patient_ids=pids,
            berichte_df=berichte_df,
        )
        if spine_available
        else pd.DataFrame()
    )

    if not spine_available and not preds.empty:
        LOGGER.warning(
            "Berichte.csv not available; cohort uses prediction export only (%d rows).",
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
            "raw_spine_selected_rows": 0,
            "prediction_reports": len(preds),
            "merged_reports": len(merged),
            "exported_cohort_rows": len(merged),
            "only_in_berichte": 0,
            "only_in_predictions": 0,
            "prediction_match_rate_pct": 100.0 if len(merged) else 0.0,
        }

    if berichte.empty and not preds.empty:
        LOGGER.warning(
            "Berichte spine empty; validation cohort uses prediction rows only (%d).",
            len(preds),
        )
        merged = preds.copy()
        stats = {
            "berichte_reports": 0,
            "raw_spine_selected_rows": 0,
            "prediction_reports": len(preds),
            "merged_reports": len(merged),
            "exported_cohort_rows": len(merged),
            "only_in_berichte": 0,
            "only_in_predictions": len(preds),
            "prediction_match_rate_pct": 0.0,
        }
        return apply_processing_fields(merged.drop(columns=["_has_prediction_row"], errors="ignore")), stats

    if berichte.empty:
        return pd.DataFrame(), {
            "berichte_reports": 0,
            "raw_spine_selected_rows": 0,
            "prediction_reports": 0,
            "merged_reports": 0,
            "exported_cohort_rows": 0,
            "only_in_berichte": 0,
            "only_in_predictions": 0,
            "prediction_match_rate_pct": 0.0,
        }

    raw_spine_rows = len(berichte)
    merged, merge_strategy, merge_keys = _merge_predictions_onto_spine(berichte, preds)

    for col, default in PREDICTION_FILL_DEFAULTS.items():
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default)

    only_berichte = int((~merged["_has_prediction_row"]).sum())
    prediction_matched = int(merged["_has_prediction_row"].sum())
    match_rate = (100.0 * prediction_matched / raw_spine_rows) if raw_spine_rows else 0.0
    missing_reasons = diagnose_missing_prediction_reasons(berichte, preds, merged)

    only_preds = 0
    if not preds.empty and not berichte.empty:
        pred_keys = [k for k in FALLBACK_MERGE_KEYS if k in preds.columns]
        ber_keys = [k for k in FALLBACK_MERGE_KEYS if k in berichte.columns]
        join_keys = [k for k in pred_keys if k in ber_keys]
        if join_keys:
            m = preds[pred_keys].drop_duplicates(subset=join_keys).merge(
                berichte[ber_keys].drop_duplicates(subset=join_keys),
                on=join_keys,
                how="left",
                indicator=True,
            )
            only_preds = int((m["_merge"] == "left_only").sum())

    if only_berichte:
        LOGGER.info(
            "Validation cohort: %d / %d spine rows without prediction (%.1f%% matched).",
            only_berichte,
            raw_spine_rows,
            match_rate,
        )

    stats = {
        "berichte_reports": raw_spine_rows,
        "raw_spine_selected_rows": raw_spine_rows,
        "prediction_reports": len(preds),
        "merged_reports": len(merged),
        "exported_cohort_rows": len(merged),
        "only_in_berichte": only_berichte,
        "only_in_predictions": only_preds,
        "eligible_spine_patients": int(berichte["PatientenID"].nunique()),
        "prediction_matched_reports": prediction_matched,
        "missing_prediction_reports": only_berichte,
        "prediction_match_rate_pct": round(match_rate, 2),
        "merge_strategy": merge_strategy,
        "merge_keys": merge_keys,
        "missing_match_reasons": missing_reasons,
    }
    merged = apply_processing_fields(merged)
    if "klasse" in merged.columns:
        merged["model_report_prediction"] = (
            pd.to_numeric(merged["klasse"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )
    merged = merged.drop(columns=["_has_prediction_row"], errors="ignore")
    drop_suffix = [c for c in merged.columns if c.endswith("_pred")]
    merged = merged.drop(columns=drop_suffix, errors="ignore")

    assert_spine_row_count_preserved(berichte, merged, context="build_complete_validation_reports_frame")
    if stats["raw_spine_selected_rows"] != stats["exported_cohort_rows"]:
        raise ValueError(
            f"raw_spine_selected_rows ({stats['raw_spine_selected_rows']}) != "
            f"exported_cohort_rows ({stats['exported_cohort_rows']})"
        )
    return merged, stats


def cohort_processing_summary_lines(cohort: pd.DataFrame) -> List[str]:
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
