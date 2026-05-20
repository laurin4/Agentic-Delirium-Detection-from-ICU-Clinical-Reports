"""
Patient-level aggregation from report-level predictions (Dokumentationsblatt excluded).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.berichte_filters import REPORT_TYPES_FOR_MATRIX, is_dokumentationsblatt, normalize_bertyp

REPORT_TYPE_COLUMNS = REPORT_TYPES_FOR_MATRIX

MANUAL_ANNOTATION_COLUMNS = (
    "manual_ground_truth",
    "possible_delir_flag",
    "alternative_explanation_flag",
    "differential_diagnosis",
    "manual_comment",
    "reviewer",
    "review_date",
)

ICDSC_GE4_THRESHOLD = 4


def _bin_klasse(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _bool_any(series: pd.Series) -> int:
    if series.empty:
        return 0
    s = series.astype(str).str.strip().str.lower()
    return int((s.isin(("1", "true", "yes"))).any())


def _aggregate_report_type(preds: pd.DataFrame, bertyp_label: str) -> pd.DataFrame:
    sub = preds[preds["bertyp"].map(normalize_bertyp) == bertyp_label]
    if sub.empty:
        return pd.DataFrame(columns=["PatientenID", bertyp_label, f"n_{bertyp_label.lower()}"])

    col_n = {
        "Verlaufseintrag": "n_verlaufseintrag",
        "Verlegungsbericht": "n_verlegungsbericht",
        "Austrittsbericht": "n_austrittsbericht",
    }[bertyp_label]

    grouped = (
        sub.groupby("PatientenID", as_index=False)
        .agg(
            **{
                bertyp_label: ("klasse", lambda s: int(_bin_klasse(s).max())),
                col_n: ("klasse", "count"),
            }
        )
    )
    return grouped


def ensure_baseline_icdsc_ge_4_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ``baseline_icdsc_ge_4`` exists for patient-level sampling and exports.

    Prefer values from ``structured_baseline`` when present; fill gaps from
    ``ICDSC_max`` (or ``max_icdsc``) using threshold >= 4.
    """
    out = df.copy()
    icdsc_col = "ICDSC_max" if "ICDSC_max" in out.columns else ("max_icdsc" if "max_icdsc" in out.columns else None)
    if icdsc_col is None:
        if "baseline_icdsc_ge_4" not in out.columns:
            out["baseline_icdsc_ge_4"] = 0
        else:
            out["baseline_icdsc_ge_4"] = (
                pd.to_numeric(out["baseline_icdsc_ge_4"], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
        return out

    derived = (pd.to_numeric(out[icdsc_col], errors="coerce").fillna(0) >= ICDSC_GE4_THRESHOLD).astype(int)
    if "baseline_icdsc_ge_4" not in out.columns:
        out["baseline_icdsc_ge_4"] = derived
    else:
        existing = pd.to_numeric(out["baseline_icdsc_ge_4"], errors="coerce")
        out["baseline_icdsc_ge_4"] = existing.fillna(derived).astype(int).clip(0, 1)
    return out


def build_patient_reporttype_matrix(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate report-level predictions to one row per PatientenID."""
    pred = normalize_patient_id_column(predictions)
    if "PatientenID" not in pred.columns:
        raise ValueError("Predictions must contain PatientenID")
    if "bertyp" not in pred.columns:
        pred["bertyp"] = ""
    pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
    pred = pred[~pred["bertyp"].map(is_dokumentationsblatt)].copy()
    pred["klasse"] = _bin_klasse(pred["klasse"])

    base = normalize_patient_id_column(baseline)
    base = base.drop_duplicates(subset=["PatientenID"], keep="first")

    patient_ids = sorted(set(pred["PatientenID"].unique()) | set(base["PatientenID"].unique()))
    out = pd.DataFrame({"PatientenID": patient_ids})

    base_cols = {
        "ICDSC_max": "max_icdsc",
        "ICD10": "baseline_icd10",
        "baseline_composite": "baseline_composite",
        "baseline_icdsc_ge_4": "baseline_icdsc_ge_4",
    }
    for out_col, src in base_cols.items():
        if src in base.columns:
            merged = base[["PatientenID", src]].rename(columns={src: out_col})
            out = out.merge(merged, on="PatientenID", how="left")
        elif out_col == "baseline_icdsc_ge_4":
            out[out_col] = pd.NA
        else:
            out[out_col] = 0

    for col in ("ICDSC_max", "ICD10", "baseline_composite"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out = ensure_baseline_icdsc_ge_4_column(out)

    for rt in REPORT_TYPE_COLUMNS:
        agg = _aggregate_report_type(pred, rt)
        out = out.merge(agg, on="PatientenID", how="left")
        out[rt] = pd.to_numeric(out[rt], errors="coerce").fillna(0).astype(int)
        n_col = {
            "Verlaufseintrag": "n_verlaufseintrag",
            "Verlegungsbericht": "n_verlegungsbericht",
            "Austrittsbericht": "n_austrittsbericht",
        }[rt]
        if n_col in out.columns:
            out[n_col] = pd.to_numeric(out[n_col], errors="coerce").fillna(0).astype(int)

    if "manual_review_candidate" in pred.columns:
        rev = (
            pred.groupby("PatientenID")["manual_review_candidate"]
            .apply(_bool_any)
            .reset_index(name="any_manual_review_candidate")
        )
        out = out.merge(rev, on="PatientenID", how="left")
    else:
        out["any_manual_review_candidate"] = 0

    for flag, col in (
        ("has_direct_delir_evidence", "any_direct_delir_evidence"),
        ("has_indirect_delir_evidence", "any_indirect_delir_evidence"),
    ):
        if flag in pred.columns:
            ev = pred.groupby("PatientenID")[flag].apply(_bool_any).reset_index(name=col)
            out = out.merge(ev, on="PatientenID", how="left")
        else:
            out[col] = 0

    out["any_manual_review_candidate"] = (
        pd.to_numeric(out["any_manual_review_candidate"], errors="coerce").fillna(0).astype(int)
    )
    out["any_direct_delir_evidence"] = (
        pd.to_numeric(out["any_direct_delir_evidence"], errors="coerce").fillna(0).astype(int)
    )
    out["any_indirect_delir_evidence"] = (
        pd.to_numeric(out["any_indirect_delir_evidence"], errors="coerce").fillna(0).astype(int)
    )

    out["model_patient_positive"] = out[list(REPORT_TYPE_COLUMNS)].max(axis=1).astype(int)

    out["discrepancy_model_vs_baseline"] = (
        out["model_patient_positive"] != out["baseline_composite"]
    ).astype(int)

    for col in MANUAL_ANNOTATION_COLUMNS:
        out[col] = ""

    out["discrepancy_manual_vs_baseline"] = 0
    out["discrepancy_manual_vs_model"] = 0

    column_order = [
        "PatientenID",
        "ICDSC_max",
        "baseline_icdsc_ge_4",
        "ICD10",
        "baseline_composite",
        *REPORT_TYPE_COLUMNS,
        "n_verlaufseintrag",
        "n_verlegungsbericht",
        "n_austrittsbericht",
        "any_manual_review_candidate",
        "any_direct_delir_evidence",
        "any_indirect_delir_evidence",
        "model_patient_positive",
        "discrepancy_model_vs_baseline",
        *MANUAL_ANNOTATION_COLUMNS,
        "discrepancy_manual_vs_baseline",
        "discrepancy_manual_vs_model",
    ]
    return out[[c for c in column_order if c in out.columns]]


def recompute_discrepancies_after_manual(matrix: pd.DataFrame) -> pd.DataFrame:
    """Recompute manual discrepancy columns when manual_ground_truth is filled (0/1)."""
    df = matrix.copy()
    manual = pd.to_numeric(df.get("manual_ground_truth", ""), errors="coerce")
    has_manual = manual.notna() & manual.isin([0, 1])
    m = manual.fillna(-1).astype(int)
    df["discrepancy_manual_vs_baseline"] = 0
    df["discrepancy_manual_vs_model"] = 0
    if has_manual.any():
        df.loc[has_manual, "discrepancy_manual_vs_baseline"] = (
            m.loc[has_manual] != df.loc[has_manual, "baseline_composite"].astype(int)
        ).astype(int)
        df.loc[has_manual, "discrepancy_manual_vs_model"] = (
            m.loc[has_manual] != df.loc[has_manual, "model_patient_positive"].astype(int)
        ).astype(int)
    return df
