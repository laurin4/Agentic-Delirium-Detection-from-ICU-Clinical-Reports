"""
PRIMARY manual validation export: patient-level cohort, report-level rows.

- Prediction unit: one report = one prediction (Verlauf / Verlegung / Austritt).
- Validation unit: unique patients (default 100 via PATIENT_VALIDATION_N).
- Manual annotation: per report (manual_report_ground_truth 0/1).
- Patient-level manual GT derived automatically (derived_manual_patient_ground_truth).
- ICDSC / ICD10 are reference signals only (not absolute truth).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.cohort_counts import load_structured_baseline_rows
from src.analysis.manual_validation_eval import (
    DERIVED_PATIENT_GT_COL,
    N_POSITIVE_REPORTS_COL,
    derive_patient_manual_labels,
)
from src.analysis.patient_reporttype_matrix import (
    build_patient_reporttype_matrix,
    ensure_baseline_icdsc_ge_4_column,
)
from src.analysis.validation_cohort_reports import (
    build_complete_validation_reports_frame,
    cohort_processing_summary_lines,
    load_raw_included_report_spine,
    per_patient_spine_export_counts,
)
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL
from src.analysis.validation_ids import (
    assign_validation_patient_ids,
    format_validation_report_id,
)
from src.pipeline.baseline_composite import compute_baseline_composite
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PATIENT_VALIDATION_COHORT_PATH,
    PATIENT_VALIDATION_COHORT_REPORT_PATH,
    PREDICTIONS_DIR,
    STRUCTURED_BASELINE_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.berichte_filters import normalize_bertyp
from src.preprocessing.berichte_mapper import read_berichte_csv_robust

LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"
DEFAULT_TARGET_N = 100


def resolve_predictions_path_for_export(
    predictions_path: Optional[Path] = None,
    *,
    prefer_validation_cohort_predictions: bool = True,
) -> Path:
    """
    Prediction CSV for cohort export / merge.

    Prefers ``validation_cohort_predictions.csv`` when present (cohort-limited inference run).
    """
    if predictions_path is not None:
        return predictions_path
    if prefer_validation_cohort_predictions and VALIDATION_COHORT_PREDICTIONS_PATH.exists():
        LOGGER.info(
            "Using cohort-limited predictions: %s",
            VALIDATION_COHORT_PREDICTIONS_PATH,
        )
        return VALIDATION_COHORT_PREDICTIONS_PATH
    return DEFAULT_PREDICTIONS_PATH

REPORT_PATIENT_LEVEL_WARNING = (
    "Patient-level reference positive; this report may still be correctly negative."
)

MANUAL_ANNOTATION_COLUMNS = (
    "manual_report_ground_truth",
    "manual_report_confidence",
    "manual_possible_delir_flag",
    "manual_alternative_explanation_flag",
    "manual_differential_diagnosis",
    "manual_comment",
    "reviewer",
    "review_date",
)

COHORT_COLUMNS: List[str] = [
    "validation_patient_id",
    "validation_report_id",
    "report_nr_within_patient",
    "PatientenID",
    SOURCE_REPORT_ROW_ID_COL,
    "bericht",
    "bertyp",
    "berdat",
    "model_report_prediction",
    "signalstaerke",
    "delir_probability_estimate",
    "manual_review_candidate",
    "decision_rule_applied",
    "status",
    "llm_called",
    "skipped_reason",
    "model_patient_positive",
    "baseline_icd10",
    "baseline_icdsc_ge_4",
    "ICDSC_max",
    "baseline_composite_or",
    "baseline_composite_and",
    *MANUAL_ANNOTATION_COLUMNS,
    DERIVED_PATIENT_GT_COL,
    N_POSITIVE_REPORTS_COL,
    "evidence_snippets",
    "delir_signale",
    "kontext",
    "begruendung",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_text_reduction_method",
    "suggested_patient_sampling_group",
    "report_patient_level_warning",
    "missing_structured_baseline",
]

SAMPLING_GROUP_ORDER: Tuple[str, ...] = (
    "model_positive",
    "model_negative",
    "icdsc_reference_positive",
    "icdsc_reference_negative",
    "icd10_reference_positive",
    "icd10_reference_negative",
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


def _merge_berdat_from_berichte(predictions: pd.DataFrame, berichte_path: Path) -> pd.DataFrame:
    if "berdat" in predictions.columns and predictions["berdat"].notna().any():
        return predictions
    if not berichte_path.exists():
        predictions["berdat"] = ""
        return predictions
    try:
        ber = normalize_patient_id_column(
            read_berichte_csv_robust(berichte_path, log_context="validation merge")
        )
    except (ValueError, OSError) as exc:
        LOGGER.warning(
            "Berichte.csv could not be loaded for berdat merge; continuing without dates: %s",
            exc,
        )
        predictions["berdat"] = ""
        return predictions
    ber.columns = [str(c).strip() for c in ber.columns]
    keys = ["PatientenID", "bericht", "bertyp"]
    if not all(k in ber.columns for k in keys):
        LOGGER.warning(
            "Berichte.csv missing merge keys %s; berdat merge skipped.",
            [k for k in keys if k not in ber.columns],
        )
        predictions["berdat"] = ""
        return predictions
    if "bertyp" in ber.columns:
        ber["bertyp"] = ber["bertyp"].map(normalize_bertyp)
    ber = ber[keys + ["berdat"]].drop_duplicates(keys, keep="first")
    pred = normalize_patient_id_column(predictions.copy())
    if "bertyp" in pred.columns:
        pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
    merged = pred.merge(ber, on=keys, how="left", suffixes=("", "_src"))
    if "berdat_src" in merged.columns:
        merged["berdat"] = merged.get("berdat", merged["berdat_src"]).fillna(merged["berdat_src"])
        merged = merged.drop(columns=["berdat_src"], errors="ignore")
    if "berdat" not in merged.columns:
        merged["berdat"] = ""
    return merged


def _filter_included_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    from src.analysis.validation_cohort_reports import _filter_included_predictions as _filter

    return _filter(predictions)


def build_patient_level_sampling_frame(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    berichte_path: Optional[Path] = None,
    berichte_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    One row per eligible patient for cohort sampling.

    Patient universe = all included Berichte reports (not prediction export only).
    Predictions are left-merged onto the report spine; missing rows count as klasse=0.
    """
    from src.analysis.validation_cohort_reports import _merge_predictions_onto_spine

    bpath = berichte_path or BERICHTE_INPUT_PATH
    spine = load_raw_included_report_spine(bpath, berichte_df=berichte_df)
    stats: dict = {
        "eligible_spine_patients": int(spine["PatientenID"].nunique()) if not spine.empty else 0,
    }

    if spine.empty:
        LOGGER.warning(
            "Berichte spine empty at %s; patient sampling falls back to prediction export only.",
            bpath,
        )
        matrix = _patient_level_frame_from_predictions(predictions, baseline)
        stats["sampling_source"] = "predictions_only"
        return matrix, stats

    preds = _filter_included_predictions(predictions)
    merged, _, _ = _merge_predictions_onto_spine(spine, preds)
    if "klasse" in merged.columns:
        merged["klasse"] = (
            pd.to_numeric(merged["klasse"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )
    else:
        merged["klasse"] = 0

    matrix = build_patient_reporttype_matrix(merged, baseline)
    stats["sampling_source"] = "berichte_spine"
    return matrix, stats


def _patient_level_frame_from_predictions(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    pred = _filter_included_predictions(predictions)
    matrix = build_patient_reporttype_matrix(pred, baseline)
    if "any_manual_review_candidate" not in matrix.columns:
        if "manual_review_candidate" in pred.columns and not pred.empty:
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
    *,
    berichte_path: Optional[Path] = None,
    berichte_df: Optional[pd.DataFrame] = None,
    prefer_berichte_spine: bool = True,
) -> pd.DataFrame:
    """Patient-level frame for balanced sampling (Berichte spine when available)."""
    bpath = berichte_path or BERICHTE_INPUT_PATH
    if prefer_berichte_spine and (berichte_df is not None or bpath.exists()):
        m, _ = build_patient_level_sampling_frame(
            predictions,
            baseline,
            berichte_path=bpath,
            berichte_df=berichte_df,
        )
    elif matrix_path.exists():
        LOGGER.info("Using patient matrix at %s (Berichte spine not available).", matrix_path)
        m = normalize_patient_id_column(pd.read_csv(matrix_path))
        if "any_manual_review_candidate" not in m.columns:
            m["any_manual_review_candidate"] = 0
        m = ensure_baseline_icdsc_ge_4_column(m)
    else:
        LOGGER.info(
            "Patient matrix not found at %s; building from predictions only.",
            matrix_path,
        )
        m = _patient_level_frame_from_predictions(predictions, baseline)
    if "baseline_icd10" not in m.columns and "ICD10" in m.columns:
        m["baseline_icd10"] = m["ICD10"]
    return normalize_patient_id_column(m)


def _count_report_types_present(row: pd.Series) -> int:
    n = 0
    for col in ("n_verlaufseintrag", "n_verlegungsbericht", "n_austrittsbericht"):
        if col in row.index and _int01(row.get(col)) > 0:
            n += 1
    return n


def assign_primary_sampling_group(row: pd.Series) -> str:
    """One primary label per patient for balanced selection (reference-aware, not composite-only)."""
    model = _int01(row.get("model_patient_positive"))
    icdsc = _int01(row.get("baseline_icdsc_ge_4"))
    icd10 = _int01(row.get("baseline_icd10"))
    if model == 1:
        return "model_positive"
    if model == 0:
        return "model_negative"
    if icdsc == 1:
        return "icdsc_reference_positive"
    if icdsc == 0:
        return "icdsc_reference_negative"
    if icd10 == 1:
        return "icd10_reference_positive"
    if icd10 == 0:
        return "icd10_reference_negative"
    return "other"


def assign_sampling_groups(patient_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_baseline_icdsc_ge_4_column(patient_df.copy())
    if "baseline_icd10" not in out.columns and "ICD10" in out.columns:
        out["baseline_icd10"] = out["ICD10"]
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
    """Balanced patient selection across model / ICDSC / ICD10 / manual review / report types."""
    df = assign_sampling_groups(patient_df)
    n_buckets = 8
    per_bucket = max(3, target_n // n_buckets)
    seen: set[str] = set()
    selected: List[str] = []

    buckets: List[Tuple[str, Callable[[pd.Series], bool]]] = [
        ("model_positive", lambda r: assign_primary_sampling_group(r) == "model_positive"),
        ("model_negative", lambda r: assign_primary_sampling_group(r) == "model_negative"),
        ("icdsc_reference_positive", lambda r: _int01(r.get("baseline_icdsc_ge_4")) == 1),
        ("icdsc_reference_negative", lambda r: _int01(r.get("baseline_icdsc_ge_4")) == 0),
        ("icd10_reference_positive", lambda r: _int01(r.get("baseline_icd10")) == 1),
        ("icd10_reference_negative", lambda r: _int01(r.get("baseline_icd10")) == 0),
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


def _baseline_reference_for_patient(
    base: pd.DataFrame,
    pid: str,
) -> Tuple[int, dict]:
    """Return (missing_flag, reference dict) for one patient."""
    if base.empty or "PatientenID" not in base.columns:
        return 1, {}
    row = base[base["PatientenID"].astype(str) == str(pid)]
    if row.empty:
        return 1, {}
    br = row.iloc[0]
    ge4 = _int01(br.get("baseline_icdsc_ge_4"))
    icd10 = _int01(br.get("baseline_icd10"))
    comp_or = int(compute_baseline_composite(pd.Series([ge4]), pd.Series([icd10]), mode="OR").iloc[0])
    comp_and = int(compute_baseline_composite(pd.Series([ge4]), pd.Series([icd10]), mode="AND").iloc[0])
    return 0, {
        "baseline_icd10": icd10,
        "baseline_icdsc_ge_4": ge4,
        "ICDSC_max": br.get("max_icdsc", ""),
        "baseline_composite_or": comp_or,
        "baseline_composite_and": comp_and,
    }


def build_patient_validation_cohort(
    predictions: pd.DataFrame,
    baseline: Optional[pd.DataFrame],
    patient_context: pd.DataFrame,
    selected_patient_ids: Sequence[str],
    *,
    berichte_path: Optional[Path] = None,
    berichte_reports: Optional[pd.DataFrame] = None,
    merge_stats: Optional[dict] = None,
    raw_spine_for_assert: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """All included report rows for selected patients (raw Berichte spine + LEFT predictions)."""
    bpath = berichte_path or BERICHTE_INPUT_PATH
    pids = [str(p) for p in selected_patient_ids]
    spine_selected = raw_spine_for_assert
    if spine_selected is None:
        spine_selected = load_raw_included_report_spine(
            bpath,
            patient_ids=pids,
            berichte_df=berichte_reports,
        )

    pred, stats = build_complete_validation_reports_frame(
        predictions,
        selected_patient_ids,
        berichte_path=bpath,
        berichte_df=berichte_reports if berichte_reports is not None else spine_selected,
    )
    if merge_stats is not None:
        merge_stats.clear()
        merge_stats.update(stats)
        merge_stats["per_patient_row_check"] = per_patient_spine_export_counts(
            spine_selected, pred
        )
    pred = _merge_berdat_from_berichte(pred, bpath)

    ctx = normalize_patient_id_column(patient_context)
    ctx = ctx[ctx["PatientenID"].isin([str(p) for p in selected_patient_ids])].drop_duplicates(
        "PatientenID", keep="first"
    )

    base = (
        normalize_patient_id_column(baseline.copy()).drop_duplicates("PatientenID", keep="first")
        if baseline is not None and not baseline.empty
        else pd.DataFrame(columns=["PatientenID"])
    )

    pid_to_vpid = assign_validation_patient_ids(list(selected_patient_ids))
    rows: List[dict] = []

    for pid in selected_patient_ids:
        validation_patient_id = pid_to_vpid[str(pid)]
        patient_reports = pred[pred["PatientenID"].astype(str) == str(pid)].copy()
        sort_cols = ["berdat", "bertyp", "bericht"]
        for c in sort_cols:
            if c not in patient_reports.columns:
                patient_reports[c] = ""
        if "berdat" in patient_reports.columns:
            patient_reports["_berdat_sort"] = pd.to_datetime(
                patient_reports["berdat"], errors="coerce"
            )
            patient_reports = patient_reports.sort_values(
                ["_berdat_sort", "bertyp", "bericht"],
                kind="mergesort",
            ).drop(columns=["_berdat_sort"])
        else:
            patient_reports = patient_reports.sort_values(
                ["bertyp", "bericht"], kind="mergesort"
            )

        ctx_row = ctx[ctx["PatientenID"].astype(str) == str(pid)]
        ctx_dict = ctx_row.iloc[0].to_dict() if not ctx_row.empty else {}
        missing_base, ref = _baseline_reference_for_patient(base, str(pid))
        model_patient_pos = _int01(ctx_dict.get("model_patient_positive"))
        if not model_patient_pos and len(patient_reports):
            model_patient_pos = max(_int01(r.get("klasse")) for _, r in patient_reports.iterrows())

        sampling_group = str(
            ctx_dict.get("suggested_patient_sampling_group")
            or assign_primary_sampling_group(pd.Series({**ctx_dict, **ref, "model_patient_positive": model_patient_pos}))
        )

        for report_nr, (_, rep) in enumerate(patient_reports.iterrows(), start=1):
            if "model_report_prediction" in rep.index and pd.notna(rep.get("model_report_prediction")):
                model_pred = _int01(rep.get("model_report_prediction"))
            else:
                model_pred = _int01(rep.get("klasse"))
            warning = ""
            if not missing_base and ref.get("baseline_icdsc_ge_4") == 1 and model_pred == 0:
                warning = REPORT_PATIENT_LEVEL_WARNING
            if not missing_base and ref.get("baseline_icd10") == 1 and model_pred == 0:
                warning = warning or REPORT_PATIENT_LEVEL_WARNING

            row = {
                "validation_patient_id": validation_patient_id,
                "validation_report_id": format_validation_report_id(
                    validation_patient_id, report_nr
                ),
                "report_nr_within_patient": report_nr,
                "PatientenID": pid,
                SOURCE_REPORT_ROW_ID_COL: str(rep.get(SOURCE_REPORT_ROW_ID_COL) or ""),
                "bericht": str(rep.get("bericht") or ""),
                "bertyp": str(rep.get("bertyp") or ""),
                "berdat": str(rep.get("berdat") or ""),
                "model_report_prediction": model_pred,
                "signalstaerke": str(rep.get("signalstaerke") or ""),
                "delir_probability_estimate": rep.get("delir_probability_estimate", ""),
                "manual_review_candidate": rep.get("manual_review_candidate", ""),
                "decision_rule_applied": str(rep.get("decision_rule_applied") or ""),
                "status": str(rep.get("status") or ""),
                "llm_called": _int01(rep.get("llm_called")),
                "skipped_reason": str(rep.get("skipped_reason") or ""),
                "model_patient_positive": model_patient_pos,
                **ref,
                "evidence_snippets": rep.get("evidence_snippets", ""),
                "delir_signale": rep.get("delir_signale", ""),
                "kontext": rep.get("kontext", ""),
                "begruendung": rep.get("begruendung", ""),
                "original_report_text_length": rep.get("original_report_text_length", ""),
                "llm_report_text_length": rep.get("llm_report_text_length", ""),
                "llm_text_reduction_method": rep.get("llm_text_reduction_method", ""),
                "suggested_patient_sampling_group": sampling_group,
                "report_patient_level_warning": warning,
                "missing_structured_baseline": missing_base,
            }
            for col in MANUAL_ANNOTATION_COLUMNS:
                row[col] = ""
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=COHORT_COLUMNS)

    out = pd.DataFrame(rows)
    out = derive_patient_manual_labels(out)
    out = out.sort_values(
        ["validation_patient_id", "berdat", "bertyp", "validation_report_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    ref_cols = ("ICDSC_max", "baseline_icd10", "baseline_icdsc_ge_4", "baseline_composite_or", "baseline_composite_and")
    for col in ref_cols:
        if col in out.columns:
            out[col] = out[col].astype(object)
            out.loc[out["missing_structured_baseline"] == 1, col] = ""

    return out[[c for c in COHORT_COLUMNS if c in out.columns]]


def format_cohort_report(
    cohort: pd.DataFrame,
    selected_n: int,
    *,
    merge_stats: Optional[dict] = None,
    sampling_stats: Optional[dict] = None,
) -> str:
    exported_patients = cohort["validation_patient_id"].nunique() if not cohort.empty else 0
    raw_rows = merge_stats.get("raw_spine_selected_rows", merge_stats.get("berichte_reports", 0)) if merge_stats else 0
    exported_rows = merge_stats.get("exported_cohort_rows", len(cohort)) if merge_stats else len(cohort)
    lines = [
        "Patient validation cohort export report",
        "=" * 44,
        f"target_unique_patients={selected_n}",
        f"exported_unique_patients={exported_patients}",
        f"raw_spine_selected_rows={raw_rows}",
        f"exported_cohort_rows={exported_rows}",
        f"row_count_match={raw_rows == exported_rows}",
    ]
    if sampling_stats:
        lines.append(f"eligible_spine_patients={sampling_stats.get('eligible_spine_patients', 0)}")
        lines.append(f"sampling_source={sampling_stats.get('sampling_source', '')}")
    if merge_stats:
        lines.append(f"prediction_matched_reports={merge_stats.get('prediction_matched_reports', 0)}")
        lines.append(f"missing_prediction_reports={merge_stats.get('missing_prediction_reports', 0)}")
        lines.append(
            f"prediction_match_rate_pct={merge_stats.get('prediction_match_rate_pct', 0)}"
        )
        if merge_stats.get("merge_strategy"):
            lines.append(
                f"prediction_merge_strategy={merge_stats.get('merge_strategy')} "
                f"keys={merge_stats.get('merge_keys', '')}"
            )
        reasons = merge_stats.get("missing_match_reasons") or {}
        if reasons:
            lines.append("missing_match_reasons (heuristic):")
            for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {cnt}")
        ppc = merge_stats.get("per_patient_row_check")
        if isinstance(ppc, pd.DataFrame) and not ppc.empty:
            lines.extend(["", "Per-patient raw vs exported row counts:", "-" * 44])
            for pid, row in ppc.iterrows():
                raw_n = int(row.get("raw_included_reports", 0))
                exp_n = int(row.get("exported_reports", 0))
                ok = bool(row.get("match", raw_n == exp_n))
                lines.append(
                    f"  PatientenID={pid}: raw={raw_n} exported={exp_n} "
                    f"[{'OK' if ok else 'MISMATCH'}]"
                )
    lines.extend(
        [
            "",
            "Report type distribution (rows):",
        ]
    )
    if not cohort.empty and "bertyp" in cohort.columns:
        for bt, cnt in cohort["bertyp"].value_counts().sort_index().items():
            lines.append(f"  {bt}: {cnt}")
    if not cohort.empty:
        pat = cohort.drop_duplicates("validation_patient_id")
        mp = pd.to_numeric(pat["model_patient_positive"], errors="coerce")
        lines.append(f"\nmodel_patient_positive_patients={int((mp == 1).sum())}")
        if "baseline_icdsc_ge_4" in pat.columns:
            icdsc = pd.to_numeric(pat["baseline_icdsc_ge_4"], errors="coerce")
            lines.append(f"icdsc_reference_positive_patients={int((icdsc == 1).sum())}")
        if "baseline_icd10" in pat.columns:
            icd10 = pd.to_numeric(pat["baseline_icd10"], errors="coerce")
            lines.append(f"icd10_reference_positive_patients={int((icd10 == 1).sum())}")
        if "suggested_patient_sampling_group" in pat.columns:
            lines.append("\nSampling group counts (patients):")
            for name, cnt in pat["suggested_patient_sampling_group"].value_counts().sort_index().items():
                lines.append(f"  {name}: {cnt}")
    lines.extend(
        [
            "",
            "Validation architecture",
            "-" * 44,
            "- Prediction unit: one report = one model prediction (klasse -> model_report_prediction).",
            "- Validation unit: unique patients; cohort exports ALL included reports per patient.",
            "- Included bertyp: Verlaufseintrag, Verlegungsbericht, Austrittsbericht.",
            "- Excluded: Dokumentationsblatt.",
            "- Chronological order within patient: berdat, bertyp, validation_report_id.",
            "",
            "Manual annotation (PRIMARY): manual_report_ground_truth per report (0/1).",
            "  Meaning: clinically plausible delir evidence in THIS report.",
            "- Do NOT fill manual_patient_ground_truth manually.",
            "- derived_manual_patient_ground_truth = max(manual_report_ground_truth) per patient",
            "  (filled after annotation; re-run evaluate_manual_validation).",
            "",
            "ICDSC and ICD10 are REFERENCE SIGNALS only — not absolute ground truth.",
            "- baseline_composite_or / baseline_composite_and: exploratory baselines.",
            "",
            "Evaluation after annotation:",
            "  python -m src.analysis.evaluate_manual_validation",
            "",
            "Skipped / prefilter-negative rows are included (model decision: no delir evidence).",
        ]
    )
    if merge_stats:
        lines.append(f"berichte_spine_reports={merge_stats.get('berichte_reports', 0)}")
        lines.append(f"prediction_export_reports={merge_stats.get('prediction_reports', 0)}")
        lines.append(f"reports_only_in_berichte={merge_stats.get('only_in_berichte', 0)}")
        lines.append(f"reports_only_in_predictions={merge_stats.get('only_in_predictions', 0)}")
    lines.extend(cohort_processing_summary_lines(cohort))
    return "\n".join(lines) + "\n"


def main(
    predictions_path: Optional[Path] = None,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    matrix_path: Path = PATIENT_REPORTTYPE_MATRIX_PATH,
    output_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    report_path: Path = PATIENT_VALIDATION_COHORT_REPORT_PATH,
    *,
    frozen_cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    use_frozen_cohort_patients: bool = True,
) -> None:
    pred_path = resolve_predictions_path_for_export(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions missing: {pred_path}. "
            "Run python -m src.pipeline.run_pipeline or "
            "VALIDATION_COHORT_ONLY=true python -m src.pipeline.run_pipeline."
        )

    target_n = patient_validation_n()
    preds = pd.read_csv(pred_path)

    frozen_selected_ids: Optional[List[str]] = None
    if use_frozen_cohort_patients and frozen_cohort_path.exists():
        frozen = normalize_patient_id_column(pd.read_csv(frozen_cohort_path))
        frozen_selected_ids = sorted(frozen["PatientenID"].astype(str).unique().tolist())
        LOGGER.info(
            "Using %d patients from frozen cohort for export: %s",
            len(frozen_selected_ids),
            frozen_cohort_path,
        )

    baseline: Optional[pd.DataFrame] = None
    if baseline_path.exists():
        baseline = load_structured_baseline_rows(baseline_path)
    else:
        LOGGER.warning("Baseline missing at %s; reference fields will be empty.", baseline_path)

    base_for_ctx = baseline if baseline is not None else pd.DataFrame()
    patient_ctx, sampling_stats = build_patient_level_sampling_frame(
        preds, base_for_ctx, berichte_path=BERICHTE_INPUT_PATH
    )
    eligible = int(sampling_stats.get("eligible_spine_patients", 0))
    if eligible < target_n:
        LOGGER.warning(
            "Only %d eligible spine patients (requested %d). Export will include all eligible.",
            eligible,
            target_n,
        )
    if frozen_selected_ids is not None:
        selected_ids = frozen_selected_ids[:target_n]
    else:
        selected_ids, _ = select_validation_patient_ids(patient_ctx, target_n=target_n)
    raw_spine_selected = load_raw_included_report_spine(
        BERICHTE_INPUT_PATH, patient_ids=selected_ids
    )
    merge_stats: dict = {}
    cohort = build_patient_validation_cohort(
        preds,
        baseline,
        patient_ctx,
        selected_ids,
        berichte_path=BERICHTE_INPUT_PATH,
        berichte_reports=raw_spine_selected,
        merge_stats=merge_stats,
        raw_spine_for_assert=raw_spine_selected,
    )
    sampling_stats["exported_unique_patients"] = (
        cohort["validation_patient_id"].nunique() if not cohort.empty else 0
    )
    sampling_stats["total_report_rows"] = len(cohort)

    MANUAL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)
    report_path.write_text(
        format_cohort_report(
            cohort,
            target_n,
            merge_stats=merge_stats,
            sampling_stats=sampling_stats,
        ),
        encoding="utf-8",
    )

    print(f"Wrote patient validation cohort: {output_path}")
    print(f"predictions_source={pred_path}")
    print(f"Wrote cohort report: {report_path}")
    print(
        f"unique_patients={cohort['validation_patient_id'].nunique()} "
        f"report_rows={len(cohort)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
