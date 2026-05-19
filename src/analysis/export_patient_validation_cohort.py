"""
Export patient-level manual validation cohort: N unique patients, all included reports each.

Prediction remains report-level; cohort selection is patient-level. For each selected
patient, every Verlaufseintrag / Verlegungsbericht / Austrittsbericht prediction row
is exported (Dokumentationsblatt excluded).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.cohort_counts import load_structured_baseline_rows
from src.analysis.patient_reporttype_matrix import build_patient_reporttype_matrix
from src.pipeline.paths import (
    MANUAL_VALIDATION_DIR,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PATIENT_VALIDATION_COHORT_PATH,
    PATIENT_VALIDATION_COHORT_REPORT_PATH,
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
DEFAULT_TARGET_N = 100

REPORT_PATIENT_LEVEL_WARNING = (
    "Patient-level baseline positive; this individual report may still be correctly negative."
)

MANUAL_VALIDATION_COLUMNS = (
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

COHORT_COLUMNS: List[str] = [
    "validation_patient_id",
    "PatientenID",
    "baseline_composite",
    "ICDSC_max",
    "ICD10",
    "baseline_icdsc_ge_4",
    "model_patient_positive",
    "n_reports_included",
    "n_verlaufseintrag",
    "n_verlegungsbericht",
    "n_austrittsbericht",
    "report_row_id",
    "bericht",
    "bertyp",
    "model_report_prediction",
    "signalstaerke",
    "delir_probability_estimate",
    "manual_review_candidate",
    "decision_rule_applied",
    "evidence_snippets",
    "delir_signale",
    "kontext",
    "begruendung",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_text_reduction_method",
    *MANUAL_VALIDATION_COLUMNS,
    "suggested_patient_sampling_group",
    "report_patient_level_warning",
    "missing_structured_baseline",
]

SAMPLING_GROUP_ORDER: Tuple[str, ...] = (
    "TP_composite",
    "FN_composite",
    "FP_composite",
    "TN_composite",
    "manual_review",
    "multi_report_types",
    "other",
)


def patient_validation_n() -> int:
    """Target unique patients; override with ``PATIENT_VALIDATION_N`` (default 100)."""
    raw = os.environ.get("PATIENT_VALIDATION_N", str(DEFAULT_TARGET_N)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        LOGGER.warning("Invalid PATIENT_VALIDATION_N=%r; using %s", raw, DEFAULT_TARGET_N)
        return DEFAULT_TARGET_N


def _int01(value: object, default: int = 0) -> int:
    try:
        return int(pd.to_numeric(value, errors="coerce") or default)
    except (TypeError, ValueError):
        return default


def _bool01(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(str(value).strip().lower() in ("1", "true", "yes"))


def _filter_included_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    pred = normalize_patient_id_column(predictions.copy())
    if "bertyp" not in pred.columns:
        pred["bertyp"] = ""
    pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
    pred = pred[~pred["bertyp"].map(is_dokumentationsblatt)].copy()
    pred = pred[pred["bertyp"].isin(REPORT_TYPES_FOR_MATRIX)].copy()
    return pred


def _patient_level_frame_from_predictions(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Build patient-level summary when matrix CSV is unavailable."""
    pred = _filter_included_predictions(predictions)
    matrix = build_patient_reporttype_matrix(pred, baseline)
    if "manual_review_candidate" in pred.columns:
        rev = (
            pred.groupby("PatientenID")["manual_review_candidate"]
            .apply(lambda s: int(any(_bool01(v) for v in s)))
            .reset_index(name="any_manual_review_candidate")
        )
        matrix = matrix.merge(rev, on="PatientenID", how="left")
    else:
        matrix["any_manual_review_candidate"] = 0
    matrix["any_manual_review_candidate"] = (
        pd.to_numeric(matrix["any_manual_review_candidate"], errors="coerce").fillna(0).astype(int)
    )
    return matrix


def load_patient_level_context(
    matrix_path: Path,
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    if matrix_path.exists():
        m = normalize_patient_id_column(pd.read_csv(matrix_path))
        if "any_manual_review_candidate" not in m.columns:
            m["any_manual_review_candidate"] = 0
    else:
        LOGGER.info("Patient matrix not found at %s; building from predictions.", matrix_path)
        m = _patient_level_frame_from_predictions(predictions, baseline)
    return normalize_patient_id_column(m)


def _count_report_types_present(row: pd.Series) -> int:
    n = 0
    for col in ("n_verlaufseintrag", "n_verlegungsbericht", "n_austrittsbericht"):
        if col in row.index and _int01(row.get(col)) > 0:
            n += 1
    return n


def assign_primary_sampling_group(row: pd.Series) -> str:
    """Assign one primary group per patient (priority order)."""
    base = _int01(row.get("baseline_composite"))
    model = _int01(row.get("model_patient_positive"))
    if base == 1 and model == 1:
        return "TP_composite"
    if base == 1 and model == 0:
        return "FN_composite"
    if base == 0 and model == 1:
        return "FP_composite"
    if base == 0 and model == 0:
        return "TN_composite"
    return "other"


def assign_sampling_groups(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Primary group + flags for balanced selection."""
    out = patient_df.copy()
    out["suggested_patient_sampling_group"] = out.apply(assign_primary_sampling_group, axis=1)
    out["_manual_review"] = out.get("any_manual_review_candidate", pd.Series(0, index=out.index)).map(_bool01)
    out["_multi_report_types"] = out.apply(
        lambda r: int(_count_report_types_present(r) >= 2),
        axis=1,
    )
    return out


def _pick_patients(
    df: pd.DataFrame,
    mask: pd.Series,
    n: int,
    seen: set[str],
) -> List[str]:
    candidates = df.loc[mask, "PatientenID"].astype(str).tolist()
    out: List[str] = []
    for pid in candidates:
        if pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
        if len(out) >= n:
            break
    return out


def select_validation_patient_ids(
    patient_df: pd.DataFrame,
    *,
    target_n: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Balanced patient selection; returns (ordered patient ids, patient frame with groups).
    """
    df = assign_sampling_groups(patient_df)
    per_bucket = max(5, target_n // 6)
    seen: set[str] = set()
    selected: List[str] = []

    buckets: List[Tuple[str, Callable[[pd.Series], bool]]] = [
        ("TP_composite", lambda r: assign_primary_sampling_group(r) == "TP_composite"),
        ("FN_composite", lambda r: assign_primary_sampling_group(r) == "FN_composite"),
        ("FP_composite", lambda r: assign_primary_sampling_group(r) == "FP_composite"),
        ("TN_composite", lambda r: assign_primary_sampling_group(r) == "TN_composite"),
        ("manual_review", lambda r: bool(r.get("_manual_review"))),
        ("multi_report_types", lambda r: bool(r.get("_multi_report_types"))),
    ]

    for _name, predicate in buckets:
        mask = df.apply(predicate, axis=1)
        picked = _pick_patients(df, mask, per_bucket, seen)
        selected.extend(picked)

    if len(selected) < target_n:
        for pid in df["PatientenID"].astype(str).tolist():
            if pid not in seen:
                selected.append(pid)
                seen.add(pid)
            if len(selected) >= target_n:
                break

    selected = selected[:target_n]
    subset = df[df["PatientenID"].isin(selected)].copy()
    group_map = dict(zip(subset["PatientenID"].astype(str), subset["suggested_patient_sampling_group"]))
    ordered = sorted(selected, key=lambda p: (group_map.get(p, "other"), p))
    return ordered, subset


def _baseline_context_columns() -> Tuple[str, ...]:
    return ("max_icdsc", "baseline_icd10", "baseline_icdsc_ge_4", "baseline_composite")


def build_patient_validation_cohort(
    predictions: pd.DataFrame,
    baseline: Optional[pd.DataFrame],
    patient_context: pd.DataFrame,
    selected_patient_ids: Sequence[str],
) -> pd.DataFrame:
    """All included report rows for selected patients with patient-level context repeated."""
    pred = _filter_included_predictions(predictions)
    pred = pred[pred["PatientenID"].isin(list(selected_patient_ids))].copy()

    ctx = normalize_patient_id_column(patient_context)
    ctx = ctx[ctx["PatientenID"].isin(list(selected_patient_ids))].drop_duplicates(
        "PatientenID", keep="first"
    )

    base = (
        normalize_patient_id_column(baseline.copy()).drop_duplicates("PatientenID", keep="first")
        if baseline is not None and not baseline.empty
        else pd.DataFrame(columns=["PatientenID"])
    )

    pid_to_validation_id = {pid: f"VP{i + 1:03d}" for i, pid in enumerate(selected_patient_ids)}

    rows: List[dict] = []
    report_counter = 0

    for vp_idx, pid in enumerate(selected_patient_ids, start=1):
        validation_patient_id = pid_to_validation_id[pid]
        patient_reports = pred[pred["PatientenID"] == pid].copy()
        patient_reports = patient_reports.sort_values(
            ["bertyp", "bericht"] if "bericht" in patient_reports.columns else ["bertyp"],
            kind="mergesort",
        )

        ctx_row = ctx[ctx["PatientenID"] == pid]
        ctx_dict = ctx_row.iloc[0].to_dict() if not ctx_row.empty else {}

        base_row = base[base["PatientenID"] == pid] if not base.empty and "PatientenID" in base.columns else pd.DataFrame()
        missing_base = 1 if base_row.empty else 0

        n_verlauf = int((patient_reports["bertyp"] == "Verlaufseintrag").sum())
        n_verleg = int((patient_reports["bertyp"] == "Verlegungsbericht").sum())
        n_austritt = int((patient_reports["bertyp"] == "Austrittsbericht").sum())
        n_included = len(patient_reports)

        for _, rep in patient_reports.iterrows():
            report_counter += 1
            model_pred = _int01(rep.get("klasse"))
            base_comp = ""
            icdsc = ""
            icd10 = ""
            icdsc_ge4 = ""
            if not missing_base:
                br = base_row.iloc[0]
                base_comp = _int01(br.get("baseline_composite"))
                icdsc = br.get("max_icdsc", "")
                icd10 = br.get("baseline_icd10", "")
                icdsc_ge4 = br.get("baseline_icdsc_ge_4", "")

            warning = ""
            if not missing_base and _int01(base_comp) == 1 and model_pred == 0:
                warning = REPORT_PATIENT_LEVEL_WARNING

            row = {
                "validation_patient_id": validation_patient_id,
                "PatientenID": pid,
                "baseline_composite": base_comp,
                "ICDSC_max": icdsc,
                "ICD10": icd10,
                "baseline_icdsc_ge_4": icdsc_ge4,
                "model_patient_positive": _int01(ctx_dict.get("model_patient_positive")),
                "n_reports_included": n_included,
                "n_verlaufseintrag": n_verlauf,
                "n_verlegungsbericht": n_verleg,
                "n_austrittsbericht": n_austritt,
                "report_row_id": f"{validation_patient_id}_R{report_counter:04d}",
                "bericht": str(rep.get("bericht") or ""),
                "bertyp": str(rep.get("bertyp") or ""),
                "model_report_prediction": model_pred,
                "signalstaerke": str(rep.get("signalstaerke") or ""),
                "delir_probability_estimate": rep.get("delir_probability_estimate", ""),
                "manual_review_candidate": rep.get("manual_review_candidate", ""),
                "decision_rule_applied": str(rep.get("decision_rule_applied") or ""),
                "evidence_snippets": rep.get("evidence_snippets", ""),
                "delir_signale": rep.get("delir_signale", ""),
                "kontext": rep.get("kontext", ""),
                "begruendung": rep.get("begruendung", ""),
                "original_report_text_length": rep.get("original_report_text_length", ""),
                "llm_report_text_length": rep.get("llm_report_text_length", ""),
                "llm_text_reduction_method": rep.get("llm_text_reduction_method", ""),
                "suggested_patient_sampling_group": str(
                    ctx_dict.get("suggested_patient_sampling_group", assign_primary_sampling_group(pd.Series(ctx_dict)))
                ),
                "report_patient_level_warning": warning,
                "missing_structured_baseline": missing_base,
            }
            for col in MANUAL_VALIDATION_COLUMNS:
                row[col] = ""
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=COHORT_COLUMNS)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["suggested_patient_sampling_group", "PatientenID", "bertyp", "bericht"],
        kind="mergesort",
    ).reset_index(drop=True)

    for col in ("ICDSC_max", "ICD10", "baseline_icdsc_ge_4", "baseline_composite"):
        if col in out.columns:
            out.loc[out["missing_structured_baseline"] == 1, col] = ""

    return out[[c for c in COHORT_COLUMNS if c in out.columns]]


def format_cohort_report(cohort: pd.DataFrame, selected_n: int) -> str:
    lines = [
        "Patient validation cohort export report",
        "=" * 44,
        f"target_unique_patients={selected_n}",
        f"exported_unique_patients={cohort['PatientenID'].nunique() if not cohort.empty else 0}",
        f"total_report_rows={len(cohort)}",
        "",
        "Sampling group counts (patients):",
    ]
    if not cohort.empty and "suggested_patient_sampling_group" in cohort.columns:
        grp = cohort.drop_duplicates("PatientenID")["suggested_patient_sampling_group"].value_counts()
        for name, cnt in grp.sort_index().items():
            lines.append(f"  {name}: {cnt}")
    lines.extend(["", "Report type distribution (rows):"])
    if not cohort.empty and "bertyp" in cohort.columns:
        for bt, cnt in cohort["bertyp"].value_counts().sort_index().items():
            lines.append(f"  {bt}: {cnt}")
    if not cohort.empty and "baseline_composite" in cohort.columns:
        pat = cohort.drop_duplicates("PatientenID")
        bc = pd.to_numeric(pat["baseline_composite"], errors="coerce")
        lines.append(f"\nbaseline_composite_positive_patients={int((bc == 1).sum())}")
        mp = pd.to_numeric(pat["model_patient_positive"], errors="coerce")
        lines.append(f"model_patient_positive_patients={int((mp == 1).sum())}")
    lines.extend(
        [
            "",
            "Validation methodology",
            "-" * 44,
            "- Cohort selection is PATIENT-level (unique PatientenID).",
            "- Each selected patient includes ALL report-level prediction rows for",
            "  Verlaufseintrag, Verlegungsbericht, and Austrittsbericht (Dokumentationsblatt excluded).",
            "- manual_report_ground_truth: annotate THIS report (0/1).",
            "- manual_patient_ground_truth: after reviewing all reports, delir documented",
            "  anywhere for this patient (0/1).",
            "- Patient-level ICDSC/ICD10 baseline does not imply every report is delir-positive;",
            "  one report may be correctly negative while another is correctly positive.",
            "- Derive patient-level manual truth later as any(manual_report_ground_truth==1) if needed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    matrix_path: Path = PATIENT_REPORTTYPE_MATRIX_PATH,
    output_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    report_path: Path = PATIENT_VALIDATION_COHORT_REPORT_PATH,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions missing: {predictions_path}. Run python -m src.pipeline.run_pipeline first."
        )

    target_n = patient_validation_n()
    preds = pd.read_csv(predictions_path)

    baseline: Optional[pd.DataFrame] = None
    if baseline_path.exists():
        baseline = load_structured_baseline_rows(baseline_path)
    else:
        LOGGER.warning("Baseline missing at %s; baseline fields will be empty.", baseline_path)

    base_for_ctx = baseline if baseline is not None else pd.DataFrame()
    patient_ctx = load_patient_level_context(matrix_path, preds, base_for_ctx)
    selected_ids, _ = select_validation_patient_ids(patient_ctx, target_n=target_n)
    cohort = build_patient_validation_cohort(preds, baseline, patient_ctx, selected_ids)

    MANUAL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)
    report_path.write_text(format_cohort_report(cohort, target_n), encoding="utf-8")

    print(f"Wrote patient validation cohort: {output_path}")
    print(f"Wrote cohort report: {report_path}")
    print(f"unique_patients={cohort['PatientenID'].nunique()} report_rows={len(cohort)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
