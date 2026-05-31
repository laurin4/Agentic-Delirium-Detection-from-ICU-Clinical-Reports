"""
Primary identity for frozen validation cohort evaluation.

``validation_report_id`` is the authoritative merge key between frozen cohort,
cohort-only predictions, manual labels, and final evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

VALIDATION_REPORT_ID_COL = "validation_report_id"
VALIDATION_PATIENT_ID_COL = "validation_patient_id"

PREDICTION_FIELDS_TO_MERGE: tuple[str, ...] = (
    "klasse",
    "status",
    "llm_called",
    "skipped_reason",
    "evidence_snippets",
    "signalstaerke",
    "delir_probability_estimate",
    "decision_rule_applied",
    "manual_review_candidate",
    "delir_signale",
    "kontext",
    "begruendung",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_text_reduction_method",
    "llm_skipped_by_prefilter",
    "anzahl_treffer",
    "has_direct_delir_evidence",
    "has_indirect_delir_evidence",
    "has_negated_delir_evidence",
    "has_prophylaxis_or_risk_only",
    "has_alternative_explanation",
    "alternative_erklaerung",
    "alternative_erklaerung_keywords",
    "klassifikation",
    "klassifikation_begruendung",
    "bericht",
    "bertyp",
    "berdat",
    "PatientenID",
)


def fill_missing_predictions_as_zero_enabled() -> bool:
    return os.environ.get("VALIDATION_EVAL_FILL_MISSING_PREDICTIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _norm_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def assert_validation_report_id_unique(df: pd.DataFrame, *, context: str) -> None:
    if VALIDATION_REPORT_ID_COL not in df.columns:
        raise ValueError(f"{context}: missing column {VALIDATION_REPORT_ID_COL}")
    ids = df[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_id)
    dup = ids[ids != ""][ids.duplicated(keep=False)]
    if not dup.empty:
        raise ValueError(
            f"{context}: duplicate {VALIDATION_REPORT_ID_COL} values: "
            f"{sorted(dup.unique().tolist())}"
        )


def check_cohort_prediction_alignment(
    cohort: pd.DataFrame,
    preds: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Return (errors, warnings) for cohort ↔ predictions identity alignment.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if VALIDATION_REPORT_ID_COL not in cohort.columns:
        errors.append(f"cohort missing {VALIDATION_REPORT_ID_COL}")
        return errors, warnings
    if VALIDATION_REPORT_ID_COL not in preds.columns:
        errors.append(f"predictions missing {VALIDATION_REPORT_ID_COL}")
        return errors, warnings

    try:
        assert_validation_report_id_unique(cohort, context="cohort")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        assert_validation_report_id_unique(preds, context="predictions")
    except ValueError as exc:
        errors.append(str(exc))

    c_ids = {_norm_id(v) for v in cohort[VALIDATION_REPORT_ID_COL]}
    p_ids = {_norm_id(v) for v in preds[VALIDATION_REPORT_ID_COL]}
    c_ids.discard("")
    p_ids.discard("")

    if len(cohort) != len(preds):
        errors.append(
            f"row count mismatch: cohort={len(cohort)} predictions={len(preds)}"
        )

    missing_in_preds = sorted(c_ids - p_ids)
    extra_in_preds = sorted(p_ids - c_ids)
    if missing_in_preds:
        errors.append(
            f"predictions missing {len(missing_in_preds)} validation_report_id(s): "
            f"{missing_in_preds[:10]}"
        )
    if extra_in_preds:
        errors.append(
            f"predictions contain {len(extra_in_preds)} unknown validation_report_id(s): "
            f"{extra_in_preds[:10]}"
        )

    empty_cohort = cohort[cohort[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_id) == ""]
    if not empty_cohort.empty:
        errors.append(f"cohort has {len(empty_cohort)} rows with empty validation_report_id")

    empty_preds = preds[preds[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_id) == ""]
    if not empty_preds.empty:
        errors.append(f"predictions have {len(empty_preds)} rows with empty validation_report_id")

    return errors, warnings


def merge_predictions_by_validation_report_id(
    cohort: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    fill_missing_as_zero: Optional[bool] = None,
    log_context: str = "validation evaluation",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge model outputs onto frozen cohort rows by ``validation_report_id`` only.

    Manual label columns on *cohort* are preserved. Existing model columns on cohort
    are replaced by prediction CSV values.
    """
    if fill_missing_as_zero is None:
        fill_missing_as_zero = fill_missing_predictions_as_zero_enabled()

    errors, warnings = check_cohort_prediction_alignment(cohort, preds)
    if errors:
        if fill_missing_as_zero:
            warnings.extend(errors)
        else:
            raise ValueError(
                f"{log_context}: cohort/prediction alignment failed: " + "; ".join(errors)
            )

    out = cohort.copy()
    drop_cols = [c for c in PREDICTION_FIELDS_TO_MERGE if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    pred_cols = [VALIDATION_REPORT_ID_COL] + [
        c for c in PREDICTION_FIELDS_TO_MERGE if c in preds.columns
    ]
    pred_slim = preds[pred_cols].drop_duplicates(VALIDATION_REPORT_ID_COL, keep="first")

    merged = out.merge(
        pred_slim,
        on=VALIDATION_REPORT_ID_COL,
        how="left",
        suffixes=("", "_pred"),
    )

    if "klasse" in merged.columns:
        merged["model_report_prediction"] = pd.to_numeric(merged["klasse"], errors="coerce")

    missing_mask = merged["klasse"].isna() if "klasse" in merged.columns else pd.Series(True, index=merged.index)
    missing_ids = merged.loc[missing_mask, VALIDATION_REPORT_ID_COL].astype(str).tolist()
    if missing_ids:
        msg = (
            f"{log_context}: no prediction for {len(missing_ids)} validation_report_id(s): "
            f"{missing_ids[:10]}"
        )
        if fill_missing_as_zero:
            warnings.append(msg + " (filled model_report_prediction=0)")
            if "model_report_prediction" not in merged.columns:
                merged["model_report_prediction"] = pd.NA
            merged.loc[missing_mask, "model_report_prediction"] = 0
            if "klasse" in merged.columns:
                merged.loc[missing_mask, "klasse"] = 0
            if "status" in merged.columns:
                merged.loc[missing_mask, "status"] = merged.loc[missing_mask, "status"].fillna(
                    "missing_prediction"
                )
        else:
            warnings.append(msg)

    for _, row in merged.iterrows():
        vid = _norm_id(row.get(VALIDATION_REPORT_ID_COL))
        if not vid:
            continue
        pred_row = pred_slim[pred_slim[VALIDATION_REPORT_ID_COL].astype(str) == vid]
        if pred_row.empty:
            continue
        pr = pred_row.iloc[0]
        for field in ("PatientenID", "bertyp", "berdat"):
            if field not in merged.columns or field not in pr.index:
                continue
            cv = _norm_id(row.get(field))
            pv = _norm_id(pr.get(field))
            if cv and pv and cv != pv:
                warnings.append(
                    f"{vid}: cohort/prediction {field} mismatch '{cv}' vs '{pv}' "
                    "(merge key is validation_report_id; structural fields are diagnostic)"
                )

    return merged, warnings
