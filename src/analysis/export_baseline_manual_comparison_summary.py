"""
Human-readable comparison of manual patient labels vs structured baselines and V2 model.

Writes baseline_manual_comparison_summary.txt for report writing / discussion support.

Run:
  python -m src.analysis.export_baseline_manual_comparison_summary

Requires frozen validation cohort, manual labels, validation cohort predictions, and
structured baseline (same inputs as final manual validation evaluation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.final_manual_validation_evaluation import (
    MANUAL_GT_COL,
    build_patient_level_ground_truth,
    load_merged_frozen_cohort,
    primary_evaluation_cohort,
)
from src.pipeline.paths import (
    BASELINE_MANUAL_COMPARISON_SUMMARY_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    STRUCTURED_BASELINE_PATH,
)
from src.pipeline.prompt_run_paths import (
    get_prompt_version_from_env,
    get_versioned_final_eval_dir,
    resolve_validation_predictions_path,
)

LOGGER = logging.getLogger(__name__)

# Candidate column names tried in order (first match wins).
COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "patient_id": ("validation_patient_id", "PatientenID", "patient_id", "PatientID"),
    "hospital_id": ("PatientenID", "patient_id", "PatientID"),
    "manual": (
        "derived_manual_patient_ground_truth",
        "manual_patient_ground_truth",
        "manual_patient_positive",
        "manual_gt",
    ),
    "icdsc": ("baseline_icdsc_ge_4", "icdsc_ge_4", "icdsc_positive", "ICDSC_ge_4"),
    "icd10": ("baseline_icd10", "icd10_positive", "has_delir_icd10"),
    "composite_or": ("baseline_composite_or", "composite_or"),
    "composite_and": ("baseline_composite_and", "composite_and"),
    "v2": (
        "model_patient_positive",
        "v2_patient_positive",
        "model_positive",
        "klasse_patient",
    ),
}


@dataclass(frozen=True)
class ResolvedColumns:
    patient_id: str
    hospital_id: Optional[str]
    manual: str
    icdsc: str
    icd10: str
    composite_or: str
    composite_and: str
    v2: str

    def as_mapping(self) -> Dict[str, str]:
        return {
            "patient_id": self.patient_id,
            "hospital_id": self.hospital_id or "",
            "manual": self.manual,
            "icdsc": self.icdsc,
            "icd10": self.icd10,
            "composite_or": self.composite_or,
            "composite_and": self.composite_and,
            "v2": self.v2,
        }


def _pick_column(df: pd.DataFrame, role: str) -> Optional[str]:
    for name in COLUMN_CANDIDATES[role]:
        if name in df.columns:
            return name
    return None


def resolve_comparison_columns(df: pd.DataFrame) -> Tuple[Optional[ResolvedColumns], List[str]]:
    """
    Map logical roles to actual dataframe columns.

    Returns (resolved, errors). When errors is non-empty, resolved is None.
    """
    missing_roles: List[str] = []
    picked: Dict[str, Optional[str]] = {}
    for role in COLUMN_CANDIDATES:
        picked[role] = _pick_column(df, role)

    required = ("patient_id", "manual", "icdsc", "icd10", "composite_or", "composite_and", "v2")
    for role in required:
        if not picked[role]:
            missing_roles.append(role)

    if missing_roles:
        available = ", ".join(sorted(df.columns.astype(str)))
        errors = [
            f"Could not map required role(s): {', '.join(missing_roles)}.",
            f"Available columns: {available}",
            "Update COLUMN_CANDIDATES in export_baseline_manual_comparison_summary.py "
            "or rename columns in the patient-level table.",
        ]
        return None, errors

    return (
        ResolvedColumns(
            patient_id=picked["patient_id"],  # type: ignore[arg-type]
            hospital_id=picked["hospital_id"],
            manual=picked["manual"],  # type: ignore[arg-type]
            icdsc=picked["icdsc"],  # type: ignore[arg-type]
            icd10=picked["icd10"],  # type: ignore[arg-type]
            composite_or=picked["composite_or"],  # type: ignore[arg-type]
            composite_and=picked["composite_and"],  # type: ignore[arg-type]
            v2=picked["v2"],  # type: ignore[arg-type]
        ),
        [],
    )


def _binary_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")


def _format_patient_label(row: pd.Series, cols: ResolvedColumns) -> str:
    vpid = str(row.get(cols.patient_id, "")).strip()
    if cols.hospital_id and cols.hospital_id != cols.patient_id:
        hid = str(row.get(cols.hospital_id, "")).strip()
        if hid and hid.lower() not in ("nan", "none", ""):
            return f"{vpid} (hospital ID: {hid})"
    return vpid


def _patient_id_list(subset: pd.DataFrame, cols: ResolvedColumns) -> List[str]:
    if subset.empty:
        return []
    labels = [_format_patient_label(subset.loc[idx], cols) for idx in subset.index]
    return sorted(dict.fromkeys(labels))


def _section(
    title: str,
    explanation: str,
    subset: pd.DataFrame,
    cols: ResolvedColumns,
    interpretation: str,
) -> List[str]:
    ids = _patient_id_list(subset, cols)
    lines = [
        title,
        "=" * len(title),
        "",
        "What this means",
        explanation,
        "",
        f"Count: {len(ids)}",
        "",
    ]
    if ids:
        lines.append("Patient / case IDs:")
        for label in ids:
            lines.append(f"  - {label}")
    else:
        lines.append("Patient / case IDs: (none)")
    lines.extend(["", "Interpretation", interpretation, ""])
    return lines


def build_baseline_manual_comparison_summary(
    patient_gt: pd.DataFrame,
    *,
    prompt_version: str = "v2",
    predictions_source: Optional[Path] = None,
    baseline_source: Optional[Path] = None,
) -> str:
    """Build the full human-readable summary text."""
    cols, errors = resolve_comparison_columns(patient_gt)
    if cols is None:
        return "\n".join(
            [
                "BASELINE vs MANUAL COMPARISON — COLUMN MAPPING FAILED",
                "",
                *errors,
            ]
        ) + "\n"

    complete = primary_evaluation_cohort(patient_gt)
    if complete.empty:
        return (
            "BASELINE vs MANUAL COMPARISON SUMMARY\n\n"
            "No patients with complete manual labels were found. "
            "Annotate all reports in the frozen validation cohort before running this summary.\n"
        )

    work = complete.copy()
    manual = _binary_col(work, cols.manual)
    icdsc = _binary_col(work, cols.icdsc)
    icd10 = _binary_col(work, cols.icd10)
    comp_or = _binary_col(work, cols.composite_or)
    comp_and = _binary_col(work, cols.composite_and)
    v2 = _binary_col(work, cols.v2)

    valid = (
        manual.isin([0, 1])
        & icdsc.isin([0, 1])
        & icd10.isin([0, 1])
        & comp_or.isin([0, 1])
        & comp_and.isin([0, 1])
        & v2.isin([0, 1])
    )
    work = work.loc[valid].copy()
    manual = manual.loc[valid]
    icdsc = icdsc.loc[valid]
    icd10 = icd10.loc[valid]
    comp_or = comp_or.loc[valid]
    comp_and = comp_and.loc[valid]
    v2 = v2.loc[valid]

    n_complete = len(work)
    n_manual_pos = int((manual == 1).sum())
    n_manual_neg = int((manual == 0).sum())

    missed_icdsc = work.loc[(manual == 1) & (icdsc == 0)]
    missed_icd10 = work.loc[(manual == 1) & (icd10 == 0)]
    v2_found_all_baseline_miss = work.loc[
        (manual == 1)
        & (v2 == 1)
        & (icdsc == 0)
        & (icd10 == 0)
        & (comp_or == 0)
        & (comp_and == 0)
    ]
    icdsc_fp = work.loc[(icdsc == 1) & (manual == 0)]
    icd10_fp = work.loc[(icd10 == 1) & (manual == 0)]

    mapping = cols.as_mapping()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = [
        "BASELINE vs MANUAL COMPARISON SUMMARY",
        "Delirium detection — frozen validation cohort (patient level)",
        "",
        f"Generated: {ts}",
        f"Prompt / model version: {prompt_version.upper()} (column: {cols.v2})",
        f"Patients evaluated: {n_complete} (complete manual labels only)",
        f"Manual positive patients: {n_manual_pos}",
        f"Manual negative patients: {n_manual_neg}",
        "",
        "Column mapping (auto-detected)",
        "-" * 28,
        f"  Manual reference label  -> {cols.manual}",
        f"  ICDSC baseline (>=4)    -> {cols.icdsc}",
        f"  ICD10 baseline          -> {cols.icd10}",
        f"  Composite OR baseline   -> {cols.composite_or}",
        f"  Composite AND baseline  -> {cols.composite_and}",
        f"  V2 model prediction     -> {cols.v2}",
        f"  Patient / case ID       -> {cols.patient_id}",
    ]
    if cols.hospital_id:
        lines.append(f"  Hospital patient ID     -> {cols.hospital_id}")
    if predictions_source:
        lines.append(f"Predictions source: {predictions_source}")
    if baseline_source:
        lines.append(f"Structured baseline source: {baseline_source}")
    lines.extend(
        [
            "",
            "How to read this file",
            "-" * 20,
            "Manual labels come from expert review of ICU reports. A patient is manual-positive",
            "if at least one report was labeled as clinically plausible delirium.",
            "Structured baselines use ICDSC (score >= 4) and ICD-10 delirium codes from the",
            "hospital database. OR = positive if either signal is positive; AND = positive only",
            "if both are positive. The V2 model aggregates report-level LLM predictions to",
            "patient level (positive if any report is positive).",
            "",
            "Only patients with all reports manually labeled are included.",
            "",
        ]
    )

    lines.extend(
        _section(
            "1. Manual positives missed by ICDSC",
            "Patients judged delirium-positive in manual review, but ICDSC score did not reach "
            "the positive threshold (ICDSC < 4 or no qualifying score). These cases suggest "
            "delirium documented in clinical text that structured screening did not flag.",
            missed_icdsc,
            cols,
            _interpret_missed_baseline(
                len(missed_icdsc),
                n_manual_pos,
                "ICDSC",
            ),
        )
    )
    lines.extend(
        _section(
            "2. Manual positives missed by ICD10",
            "Patients judged delirium-positive in manual review, but no qualifying ICD-10 "
            "delirium code was recorded in structured data. Documentation in reports may "
            "precede or differ from coded diagnoses.",
            missed_icd10,
            cols,
            _interpret_missed_baseline(
                len(missed_icd10),
                n_manual_pos,
                "ICD-10",
            ),
        )
    )
    lines.extend(
        _section(
            "3. Manual positives found by V2 but missed by all baseline rules",
            f"Patients where manual review and the {prompt_version.upper()} model agree on "
            "delirium presence, but ICDSC, ICD10, OR, and AND baselines are all negative. "
            "These are the clearest examples of signal recovered from report text beyond "
            "structured reference standards.",
            v2_found_all_baseline_miss,
            cols,
            _interpret_v2_beyond_baselines(len(v2_found_all_baseline_miss), n_manual_pos),
        )
    )
    lines.extend(
        _section(
            "4. False positives from ICDSC",
            "Patients flagged positive by ICDSC (score >= 4) but judged delirium-negative "
            "in manual review. May reflect screening sensitivity, transient confusion, or "
            "mismatch between screening scores and clinical documentation.",
            icdsc_fp,
            cols,
            _interpret_baseline_fp(len(icdsc_fp), n_manual_neg, "ICDSC"),
        )
    )
    lines.extend(
        _section(
            "5. False positives from ICD10",
            "Patients with a qualifying ICD-10 delirium code but judged delirium-negative "
            "in manual report review. May reflect coding timing, rule-out diagnoses, or "
            "differences between coded diagnoses and what appears in individual reports.",
            icd10_fp,
            cols,
            _interpret_baseline_fp(len(icd10_fp), n_manual_neg, "ICD-10"),
        )
    )

    lines.extend(
        [
            "OVERALL DISCUSSION NOTES",
            "======================",
            "",
            _overall_interpretation(
                n_complete=n_complete,
                n_manual_pos=n_manual_pos,
                n_missed_icdsc=len(missed_icdsc),
                n_missed_icd10=len(missed_icd10),
                n_v2_beyond=len(v2_found_all_baseline_miss),
                n_icdsc_fp=len(icdsc_fp),
                n_icd10_fp=len(icd10_fp),
                prompt_version=prompt_version,
            ),
            "",
            "Technical note: column roles resolved as "
            + ", ".join(f"{k}={v}" for k, v in mapping.items() if v),
        ]
    )
    return "\n".join(lines) + "\n"


def _interpret_missed_baseline(n_miss: int, n_manual_pos: int, baseline_name: str) -> str:
    if n_manual_pos == 0:
        return f"No manual-positive patients in this cohort; {baseline_name} misses cannot be assessed."
    share = 100.0 * n_miss / n_manual_pos
    if n_miss == 0:
        return (
            f"Every manual-positive patient was also flagged by {baseline_name}. "
            "Structured data did not miss any manually confirmed cases in this set."
        )
    return (
        f"{baseline_name} missed {n_miss} of {n_manual_pos} manual-positive patients "
        f"({share:.1f}%). This gap highlights cases where report-based review found "
        f"delirium that {baseline_name} alone would not have captured."
    )


def _interpret_v2_beyond_baselines(n_cases: int, n_manual_pos: int) -> str:
    if n_manual_pos == 0:
        return "No manual-positive patients; this category is empty."
    if n_cases == 0:
        return (
            "No patients were positive on both manual review and V2 while all structured "
            "baselines stayed negative. Where V2 and manual agree, structured signals "
            "usually also fire at least one rule."
        )
    share = 100.0 * n_cases / n_manual_pos
    return (
        f"{n_cases} manual-positive patients ({share:.1f}% of manual positives) were "
        "detected by V2 but not by any structured baseline rule. These cases support the "
        "value of report-text analysis for finding delirium beyond ICDSC and ICD-10 alone."
    )


def _interpret_baseline_fp(n_fp: int, n_manual_neg: int, baseline_name: str) -> str:
    if n_manual_neg == 0:
        return f"No manual-negative patients; {baseline_name} false positives cannot be assessed."
    share = 100.0 * n_fp / n_manual_neg
    if n_fp == 0:
        return (
            f"No {baseline_name} positives conflicted with manual-negative labels. "
            "Structured positives align with manual review among negatives in this set."
        )
    return (
        f"{baseline_name} flagged {n_fp} manual-negative patients ({share:.1f}% of manual "
        "negatives). Review these IDs when discussing specificity and the clinical meaning "
        "of structured positives versus report-level documentation."
    )


def _overall_interpretation(
    *,
    n_complete: int,
    n_manual_pos: int,
    n_missed_icdsc: int,
    n_missed_icd10: int,
    n_v2_beyond: int,
    n_icdsc_fp: int,
    n_icd10_fp: int,
    prompt_version: str,
) -> str:
    parts = [
        f"This summary compares expert manual labels ({n_complete} complete patients, "
        f"{n_manual_pos} manual-positive) against structured ICDSC/ICD-10 baselines and "
        f"the {prompt_version.upper()} report-text model.",
    ]
    if n_missed_icdsc or n_missed_icd10:
        parts.append(
            "Structured baselines do not cover all manually confirmed delirium cases; "
            f"ICDSC missed {n_missed_icdsc} and ICD-10 missed {n_missed_icd10} manual positives."
        )
    else:
        parts.append(
            "In this cohort, every manual-positive patient was also captured by both ICDSC and ICD-10."
        )
    if n_v2_beyond:
        parts.append(
            f"The {prompt_version.upper()} model identified {n_v2_beyond} manual-positive "
            "patient(s) that no structured baseline rule flagged — useful evidence for "
            "report-text signal beyond administrative data."
        )
    if n_icdsc_fp or n_icd10_fp:
        parts.append(
            f"Structured false positives (manual-negative): ICDSC {n_icdsc_fp}, ICD-10 {n_icd10_fp}. "
            "Use these lists when discussing limits of ICD-based reference standards."
        )
    else:
        parts.append("No structured false positives among manual-negative patients in this cohort.")
    parts.append(
        "Manual labels are the primary reference for validation; baselines are exploratory "
        "comparison signals, not ground truth."
    )
    return " ".join(parts)


def export_baseline_manual_comparison_summary(
    patient_gt: pd.DataFrame,
    output_path: Path = BASELINE_MANUAL_COMPARISON_SUMMARY_PATH,
    *,
    prompt_version: Optional[str] = None,
    predictions_source: Optional[Path] = None,
    baseline_source: Optional[Path] = None,
) -> Path:
    version = (prompt_version or get_prompt_version_from_env()).upper()
    text = build_baseline_manual_comparison_summary(
        patient_gt,
        prompt_version=version,
        predictions_source=predictions_source,
        baseline_source=baseline_source,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    LOGGER.info("Wrote %s", output_path)
    return output_path


def build_patient_table_from_frozen_cohort(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    predictions_path: Path | None = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    resolved_predictions = (
        predictions_path
        if predictions_path is not None
        else resolve_validation_predictions_path()
    )
    merged, resolved_baseline, _ = load_merged_frozen_cohort(
        cohort_path,
        labels_path,
        baseline_path,
        resolved_predictions,
    )
    return build_patient_level_ground_truth(merged), resolved_predictions, resolved_baseline


def main(
    output_path: Path | None = None,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    predictions_path: Path | None = None,
) -> None:
    resolved_output = output_path
    if resolved_output is None:
        try:
            resolved_output = get_versioned_final_eval_dir() / "baseline_manual_comparison_summary.txt"
        except ValueError:
            resolved_output = BASELINE_MANUAL_COMPARISON_SUMMARY_PATH

    patient_gt, pred_src, base_src = build_patient_table_from_frozen_cohort(
        cohort_path=cohort_path,
        labels_path=labels_path,
        baseline_path=baseline_path,
        predictions_path=predictions_path,
    )
    path = export_baseline_manual_comparison_summary(
        patient_gt,
        output_path=resolved_output,
        predictions_source=pred_src,
        baseline_source=base_src,
    )
    print(path.read_text(encoding="utf-8"))
    print(f"Saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
