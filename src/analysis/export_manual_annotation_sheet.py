"""
Export report-level manual annotation sheet with patient-level baseline context.

One row per processed report prediction (Dokumentationsblatt excluded).
Manual labels are report-level first; patient-level truth can be derived later
as max(manual_report_ground_truth) per patient — do not copy one patient label
onto every report row.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.analysis.cohort_counts import load_structured_baseline_rows
from src.pipeline.baseline_composite import baseline_composite_definition_text
from src.pipeline.paths import (
    MANUAL_ANNOTATION_SHEET_PATH,
    MANUAL_ANNOTATION_SHEET_REPORT_PATH,
    MANUAL_VALIDATION_DIR,
    PREDICTIONS_DIR,
    STRUCTURED_BASELINE_PATH,
)
from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.berichte_filters import (
    REPORT_TYPES_FOR_MATRIX,
    is_dokumentationsblatt,
    normalize_bertyp,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"

REPORT_PATIENT_LEVEL_WARNING = (
    "Baseline is patient-level; this report may not contain delirium even if "
    "patient baseline is positive."
)

BASELINE_MERGE_COLUMNS = (
    "max_icdsc",
    "baseline_icd10",
    "baseline_icdsc_ge_4",
    "baseline_composite",
)

MANUAL_ANNOTATION_COLUMNS = (
    "manual_report_ground_truth",
    "manual_patient_ground_truth",
    "manual_possible_delir_flag",
    "manual_alternative_explanation_flag",
    "manual_differential_diagnosis",
    "manual_discrepancy_type",
    "manual_comment",
    "reviewer",
    "review_date",
)

ANNOTATION_SHEET_COLUMNS: List[str] = [
    "annotation_row_id",
    "PatientenID",
    "bericht",
    "bertyp",
    "model_report_prediction",
    "model_klassifikation",
    "signalstaerke",
    "delir_probability_estimate",
    "manual_review_candidate",
    "decision_rule_applied",
    "evidence_snippets",
    "delir_signale",
    "kontext",
    "begruendung",
    "ICDSC_max",
    "ICD10",
    "baseline_icdsc_ge_4",
    "baseline_composite",
    "model_any_verlaufseintrag",
    "model_any_verlegungsbericht",
    "model_any_austrittsbericht",
    "model_patient_positive",
    *MANUAL_ANNOTATION_COLUMNS,
    "model_vs_composite_baseline_discrepancy",
    "model_patient_vs_composite_baseline_discrepancy",
    "report_patient_level_warning",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_text_reduction_method",
    "llm_skipped_by_prefilter",
    "has_direct_delir_evidence",
    "has_indirect_delir_evidence",
    "has_negated_delir_evidence",
    "has_prophylaxis_or_risk_only",
    "has_alternative_explanation",
    "missing_structured_baseline",
]


def _bin_klasse(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _patient_model_aggregations(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-patient max prediction per report type and overall patient positive."""
    work = pred[["PatientenID", "bertyp", "model_report_prediction"]].copy()
    rows: List[dict] = []
    for pid, grp in work.groupby("PatientenID", sort=False):
        row: dict = {"PatientenID": pid}
        type_max = {}
        for rt in REPORT_TYPES_FOR_MATRIX:
            sub = grp[grp["bertyp"] == rt]
            val = int(sub["model_report_prediction"].max()) if not sub.empty else 0
            col = {
                "Verlaufseintrag": "model_any_verlaufseintrag",
                "Verlegungsbericht": "model_any_verlegungsbericht",
                "Austrittsbericht": "model_any_austrittsbericht",
            }[rt]
            row[col] = val
            type_max[rt] = val
        row["model_patient_positive"] = max(type_max.values()) if type_max else 0
        rows.append(row)
    return pd.DataFrame(rows)


def build_manual_annotation_sheet(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Build annotation sheet dataframe (one row per report, sorted)."""
    pred = normalize_patient_id_column(predictions.copy())
    if "PatientenID" not in pred.columns:
        raise ValueError("Predictions must contain PatientenID")

    if "bertyp" not in pred.columns:
        pred["bertyp"] = ""
    pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
    n_before = len(pred)
    pred = pred[~pred["bertyp"].map(is_dokumentationsblatt)].copy()
    if len(pred) < n_before:
        LOGGER.info("excluded_dokumentationsblatt_rows=%d", n_before - len(pred))

    if "bericht" not in pred.columns:
        pred["bericht"] = ""

    pred["model_report_prediction"] = _bin_klasse(pred["klasse"])
    pred["model_klassifikation"] = pred.get("klassifikation", "").astype(str)

    agg = _patient_model_aggregations(pred)
    pred = pred.merge(agg, on="PatientenID", how="left")

    base = normalize_patient_id_column(baseline.copy())
    base = base.drop_duplicates(subset=["PatientenID"], keep="first")
    for col in BASELINE_MERGE_COLUMNS:
        if col not in base.columns:
            base[col] = pd.NA

    base_merge = base[["PatientenID", *BASELINE_MERGE_COLUMNS]].copy()
    merged = pred.merge(base_merge, on="PatientenID", how="left", suffixes=("", "_base"))

    merged["missing_structured_baseline"] = merged["max_icdsc"].isna().astype(int)

    merged = merged.rename(
        columns={
            "max_icdsc": "ICDSC_max",
            "baseline_icd10": "ICD10",
        }
    )

    for col in ("ICDSC_max", "ICD10", "baseline_icdsc_ge_4", "baseline_composite"):
        missing_mask = merged["missing_structured_baseline"] == 1
        merged.loc[missing_mask, col] = pd.NA

    base_comp = pd.to_numeric(merged["baseline_composite"], errors="coerce")
    merged["model_vs_composite_baseline_discrepancy"] = pd.NA
    has_base = base_comp.notna()
    merged.loc[has_base, "model_vs_composite_baseline_discrepancy"] = (
        merged.loc[has_base, "model_report_prediction"].astype(int)
        != base_comp.loc[has_base].astype(int)
    ).astype(int)

    merged["model_patient_vs_composite_baseline_discrepancy"] = pd.NA
    merged.loc[has_base, "model_patient_vs_composite_baseline_discrepancy"] = (
        merged.loc[has_base, "model_patient_positive"].astype(int)
        != base_comp.loc[has_base].astype(int)
    ).astype(int)

    merged["report_patient_level_warning"] = ""
    warn_mask = has_base & (base_comp == 1) & (merged["model_report_prediction"] == 0)
    merged.loc[warn_mask, "report_patient_level_warning"] = REPORT_PATIENT_LEVEL_WARNING

    for col in MANUAL_ANNOTATION_COLUMNS:
        merged[col] = ""

    sort_cols = ["PatientenID", "bertyp"]
    if "bericht" in merged.columns:
        sort_cols.append("bericht")
    merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    merged.insert(0, "annotation_row_id", [f"{i + 1:06d}" for i in range(len(merged))])

    out_cols = [c for c in ANNOTATION_SHEET_COLUMNS if c in merged.columns]
    extra = [c for c in merged.columns if c not in out_cols and c not in ("klasse", "klassifikation")]
    if extra:
        LOGGER.debug("Dropping non-sheet columns: %s", extra[:10])

    sheet = merged[out_cols].copy()
    for col in MANUAL_ANNOTATION_COLUMNS:
        if col in sheet.columns:
            sheet[col] = sheet[col].astype(str)

    for col in (
        "ICDSC_max",
        "ICD10",
        "baseline_icdsc_ge_4",
        "baseline_composite",
        "model_vs_composite_baseline_discrepancy",
        "model_patient_vs_composite_baseline_discrepancy",
    ):
        if col in sheet.columns:
            sheet[col] = sheet[col].astype(object)
            sheet.loc[sheet["missing_structured_baseline"] == 1, col] = ""

    return sheet


def _format_report(sheet: pd.DataFrame) -> str:
    lines: List[str] = [
        "Manual annotation sheet report",
        "=" * 40,
        f"rows={len(sheet)}",
        f"unique_patients={sheet['PatientenID'].nunique()}",
        "",
        "Report type distribution (rows):",
    ]
    if "bertyp" in sheet.columns:
        for bt, cnt in sheet["bertyp"].value_counts().sort_index().items():
            lines.append(f"  {bt}: {cnt}")
    lines.extend(["", "Model positives by report type (model_report_prediction==1):"])
    if "bertyp" in sheet.columns and "model_report_prediction" in sheet.columns:
        pos = sheet[sheet["model_report_prediction"].astype(int) == 1]
        for bt, cnt in pos["bertyp"].value_counts().sort_index().items():
            lines.append(f"  {bt}: {cnt}")
    lines.append("")
    if "baseline_composite" in sheet.columns:
        bc = pd.to_numeric(sheet["baseline_composite"], errors="coerce")
        n_comp = int((bc == 1).sum())
        lines.append(f"rows_with_baseline_composite_positive={n_comp}")
    lines.append(
        f"missing_structured_baseline_rows={int(sheet.get('missing_structured_baseline', pd.Series(0)).sum())}"
    )
    lines.extend(
        [
            "",
            "Annotation methodology",
            "-" * 40,
            "- Predictions are report-level (one row per report).",
            "- manual_report_ground_truth: annotate THIS report only (0/1).",
            "- manual_patient_ground_truth: optional patient-level label after reviewing",
            "  all included reports for that patient (0/1).",
            "- Do NOT copy one patient-level label onto every report row.",
            "- Patient-level manual truth can later be derived as:",
            "  any(manual_report_ground_truth == 1) within a patient.",
            "- Dokumentationsblatt reports are excluded from this sheet.",
            f"- {baseline_composite_definition_text()};",
            "  report_patient_level_warning flags patient-positive baseline with",
            "  model_report_prediction==0 on a single report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    output_path: Path = MANUAL_ANNOTATION_SHEET_PATH,
    report_path: Path = MANUAL_ANNOTATION_SHEET_REPORT_PATH,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions missing: {predictions_path}. Run python -m src.pipeline.run_pipeline first."
        )

    preds = pd.read_csv(predictions_path)
    baseline = load_structured_baseline_rows(baseline_path)
    sheet = build_manual_annotation_sheet(preds, baseline)

    MANUAL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output_path, index=False)

    report_text = _format_report(sheet)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote manual annotation sheet: {output_path}")
    print(f"Wrote annotation sheet report: {report_path}")
    print(f"rows={len(sheet)} patients={sheet['PatientenID'].nunique()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
