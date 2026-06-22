"""
Patient-level biostatistical evaluation for the manual validation cohort.

Read-only analysis: Wilson 95% confidence intervals for diagnostic metrics and
paired McNemar tests between selected classifiers.

Ground truth: derived_manual_patient_ground_truth (patient-level manual labels).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline.paths import (
    BIOSTATISTICS_DIR,
    CASCADE_REVIEWER_RUN_01_DIR,
    CASCADE_V1_V2_V3_RUN_01_DIR,
)
from src.pipeline.prompt_run_paths import PROMPT_RUNS_ROOT
from src.pipeline.validation_report_identity import VALIDATION_PATIENT_ID_COL

MANUAL_GT_COL = "derived_manual_patient_ground_truth"

LOGGER = logging.getLogger(__name__)

WILSON_Z = 1.959963984540054
MCNEMAR_EXACT_MAX_DISCORDANT = 25

EXPECTED_COMPLETE_PATIENTS = 100
EXPECTED_MANUAL_POSITIVES = 23
EXPECTED_MANUAL_NEGATIVES = 77

BASELINE_COLUMNS: Dict[str, str] = {
    "icdsc": "baseline_icdsc_ge_4",
    "icd10": "baseline_icd10",
    "composite_or": "baseline_composite_or",
    "composite_and": "baseline_composite_and",
}

METHOD_FILE_SPECS: Tuple[Tuple[str, Path, str], ...] = (
    (
        "v1",
        PROMPT_RUNS_ROOT / "v1" / "run_01" / "final_evaluation" / "patient_level_ground_truth.csv",
        "model_patient_positive",
    ),
    (
        "v2_run_02",
        PROMPT_RUNS_ROOT / "v2" / "run_02" / "final_evaluation" / "patient_level_ground_truth.csv",
        "model_patient_positive",
    ),
    (
        "v2_run_03",
        PROMPT_RUNS_ROOT / "v2" / "run_03" / "final_evaluation" / "patient_level_ground_truth.csv",
        "model_patient_positive",
    ),
    (
        "v2_run_04",
        PROMPT_RUNS_ROOT / "v2" / "run_04" / "final_evaluation" / "patient_level_ground_truth.csv",
        "model_patient_positive",
    ),
    (
        "cascade_standard",
        CASCADE_V1_V2_V3_RUN_01_DIR / "cascade_patient_predictions.csv",
        "cascade_patient_positive",
    ),
    (
        "cascade_reviewer",
        CASCADE_REVIEWER_RUN_01_DIR / "cascade_patient_predictions.csv",
        "cascade_patient_positive",
    ),
)

BASELINE_SOURCE_PATH = (
    PROMPT_RUNS_ROOT / "v2" / "run_02" / "final_evaluation" / "patient_level_ground_truth.csv"
)

MCNEMAR_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("v2_run_02", "icdsc"),
    ("v2_run_02", "icd10"),
    ("v2_run_02", "composite_or"),
    ("v1", "v2_run_02"),
    ("cascade_standard", "v2_run_02"),
    ("cascade_standard", "v1"),
    ("cascade_reviewer", "v2_run_02"),
    ("cascade_reviewer", "v1"),
    ("cascade_reviewer", "icdsc"),
    ("cascade_reviewer", "composite_or"),
)

METRICS_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "method",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "sensitivity",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
    "specificity",
    "specificity_ci_low",
    "specificity_ci_high",
    "PPV",
    "PPV_ci_low",
    "PPV_ci_high",
    "NPV",
    "NPV_ci_low",
    "NPV_ci_high",
    "F1",
    "accuracy",
    "accuracy_ci_low",
    "accuracy_ci_high",
)


@dataclass(frozen=True)
class PatientMethodTable:
    """Patient-level predictions for one method."""

    method_name: str
    frame: pd.DataFrame  # validation_patient_id, y_true, y_pred


def _binary_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def wilson_ci(
    successes: int,
    n: int,
    *,
    z: float = WILSON_Z,
) -> Tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (proportion, ci_low, ci_high). All NA if n == 0.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (p_hat, low, high)


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, int]:
    """TP/FP/TN/FN for aligned binary patient-level series."""
    yt = _binary_series(y_true)
    yp = _binary_series(y_pred)
    valid = yt.notna() & yp.notna() & yt.isin([0, 1]) & yp.isin([0, 1])
    yt_v = yt.loc[valid].astype(int)
    yp_v = yp.loc[valid].astype(int)

    tp = int(((yp_v == 1) & (yt_v == 1)).sum())
    tn = int(((yp_v == 0) & (yt_v == 0)).sum())
    fp = int(((yp_v == 1) & (yt_v == 0)).sum())
    fn = int(((yp_v == 0) & (yt_v == 1)).sum())
    n = tp + tn + fp + fn
    return {"n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def diagnostic_metrics_with_ci(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    method_name: str,
) -> Dict[str, Any]:
    """Compute confusion-based metrics with Wilson 95% CIs."""
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn, n = counts["TP"], counts["FP"], counts["TN"], counts["FN"], counts["n"]

    sens, sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    spec, spec_lo, spec_hi = wilson_ci(tn, tn + fp)
    ppv, ppv_lo, ppv_hi = wilson_ci(tp, tp + fp)
    npv, npv_lo, npv_hi = wilson_ci(tn, tn + fn)
    acc, acc_lo, acc_hi = wilson_ci(tp + tn, n)

    precision = ppv if not math.isnan(ppv) else 0.0
    recall = sens if not math.isnan(sens) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "method": method_name,
        "n": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": sens,
        "sensitivity_ci_low": sens_lo,
        "sensitivity_ci_high": sens_hi,
        "specificity": spec,
        "specificity_ci_low": spec_lo,
        "specificity_ci_high": spec_hi,
        "PPV": ppv,
        "PPV_ci_low": ppv_lo,
        "PPV_ci_high": ppv_hi,
        "NPV": npv,
        "NPV_ci_low": npv_lo,
        "NPV_ci_high": npv_hi,
        "F1": f1,
        "accuracy": acc,
        "accuracy_ci_low": acc_lo,
        "accuracy_ci_high": acc_hi,
    }


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def _binom_two_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    lower = _binom_cdf(k, n, p)
    upper = 1.0 - _binom_cdf(k - 1, n, p) if k > 0 else 1.0
    return min(2.0 * min(lower, upper), 1.0)


def _chi2_sf(x: float, df: int = 1) -> float:
    if x < 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    raise ValueError(f"chi2_sf only implemented for df=1, got {df}")


def mcnemar_test(b: int, c: int) -> Tuple[str, float, float]:
    """
    Paired McNemar test on discordant correctness counts.

    b = method_a correct, method_b wrong
    c = method_a wrong, method_b correct
  """
    n_discordant = b + c
    if n_discordant == 0:
        return ("none", float("nan"), float("nan"))

    if n_discordant <= MCNEMAR_EXACT_MAX_DISCORDANT:
        k = min(b, c)
        p_value = _binom_two_sided_pvalue(k, n_discordant)
        return ("exact_binomial", float(k), p_value)

    statistic = ((abs(b - c) - 1.0) ** 2) / n_discordant
    p_value = _chi2_sf(statistic, df=1)
    return ("chi_square_cc", statistic, p_value)


def _complete_patients(df: pd.DataFrame) -> pd.DataFrame:
    if "is_patient_complete" not in df.columns:
        return df.copy()
    return df[df["is_patient_complete"] == 1].copy()


def _load_method_table(
    path: Path,
    *,
    method_name: str,
    pred_col: str,
    y_true_col: str = MANUAL_GT_COL,
) -> Optional[PatientMethodTable]:
    if not path.exists():
        LOGGER.warning("Skipping method %s: file missing (%s)", method_name, path)
        return None

    df = pd.read_csv(path)
    if VALIDATION_PATIENT_ID_COL not in df.columns:
        LOGGER.warning(
            "Skipping method %s: %s missing from %s",
            method_name,
            VALIDATION_PATIENT_ID_COL,
            path,
        )
        return None
    if pred_col not in df.columns:
        LOGGER.warning(
            "Skipping method %s: prediction column %s missing from %s",
            method_name,
            pred_col,
            path,
        )
        return None
    if y_true_col not in df.columns:
        LOGGER.warning(
            "Skipping method %s: ground truth column %s missing from %s",
            method_name,
            y_true_col,
            path,
        )
        return None

    work = _complete_patients(df)
    out = pd.DataFrame(
        {
            VALIDATION_PATIENT_ID_COL: work[VALIDATION_PATIENT_ID_COL].astype(str),
            "y_true": _binary_series(work[y_true_col]),
            "y_pred": _binary_series(work[pred_col]),
        }
    )
    return PatientMethodTable(method_name=method_name, frame=out)


def load_baseline_method_tables(path: Path) -> Dict[str, PatientMethodTable]:
    """Load structured baseline predictions from a patient-level evaluation CSV."""
    tables: Dict[str, PatientMethodTable] = {}
    if not path.exists():
        LOGGER.warning("Baseline source missing: %s", path)
        return tables

    df = pd.read_csv(path)
    work = _complete_patients(df)
    if MANUAL_GT_COL not in work.columns:
        LOGGER.warning("Baseline source missing %s: %s", MANUAL_GT_COL, path)
        return tables

    for method_name, col in BASELINE_COLUMNS.items():
        if col not in work.columns:
            LOGGER.warning("Baseline column %s missing for %s in %s", col, method_name, path)
            continue
        frame = pd.DataFrame(
            {
                VALIDATION_PATIENT_ID_COL: work[VALIDATION_PATIENT_ID_COL].astype(str),
                "y_true": _binary_series(work[MANUAL_GT_COL]),
                "y_pred": _binary_series(work[col]),
            }
        )
        tables[method_name] = PatientMethodTable(method_name=method_name, frame=frame)
    return tables


def load_all_method_tables(
    method_specs: Sequence[Tuple[str, Path, str]] = METHOD_FILE_SPECS,
    baseline_source: Path = BASELINE_SOURCE_PATH,
) -> Dict[str, PatientMethodTable]:
    """Load all available patient-level method tables keyed by method name."""
    tables = load_baseline_method_tables(baseline_source)

    for method_name, path, pred_col in method_specs:
        loaded = _load_method_table(path, method_name=method_name, pred_col=pred_col)
        if loaded is not None:
            tables[method_name] = loaded
    return tables


def align_method_pair(
    table_a: PatientMethodTable,
    table_b: PatientMethodTable,
) -> pd.DataFrame:
    """Inner-join two methods on validation_patient_id with valid binary labels."""
    merged = table_a.frame.merge(
        table_b.frame,
        on=VALIDATION_PATIENT_ID_COL,
        how="inner",
        suffixes=("_a", "_b"),
    )
    valid = (
        merged["y_true_a"].notna()
        & merged["y_true_b"].notna()
        & merged["y_pred_a"].notna()
        & merged["y_pred_b"].notna()
        & merged["y_true_a"].isin([0, 1])
        & merged["y_true_b"].isin([0, 1])
        & merged["y_pred_a"].isin([0, 1])
        & merged["y_pred_b"].isin([0, 1])
    )
    work = merged.loc[valid].copy()
    if not work.empty and not (work["y_true_a"] == work["y_true_b"]).all():
        LOGGER.warning(
            "Ground truth mismatch between %s and %s on aligned patients",
            table_a.method_name,
            table_b.method_name,
        )
    work["y_true"] = work["y_true_a"].astype(int)
    return work


def discordant_odds_ratio(b: int, c: int) -> float:
    """
    Odds ratio from McNemar discordant cells (b = A correct/B wrong, c = A wrong/B correct).

    Uses Haldane-Anscombe +0.5 correction when any cell is zero.
    """
    return (b + 0.5) / (c + 0.5)


def paired_comparison_effect_sizes(
    *,
    n_common: int,
    a_correct_b_wrong: int,
    a_wrong_b_correct: int,
    both_correct: int,
) -> Dict[str, float]:
    """
    Effect sizes for a paired method comparison vs the same manual ground truth.

    accuracy_diff: paired risk difference in overall accuracy (method_a - method_b).
    discordant_odds_ratio: how often A wins vs B among discordant patients only.
    proportion_a_better_discordant: b / (b + c).
    """
    b = a_correct_b_wrong
    c = a_wrong_b_correct
    n = n_common
    if n <= 0:
        return {
            "accuracy_a": float("nan"),
            "accuracy_b": float("nan"),
            "accuracy_diff": float("nan"),
            "discordant_odds_ratio": float("nan"),
            "proportion_a_better_discordant": float("nan"),
        }

    acc_a = (both_correct + b) / n
    acc_b = (both_correct + c) / n
    discordant = b + c
    prop_a = (b / discordant) if discordant else float("nan")

    return {
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_diff": acc_a - acc_b,
        "discordant_odds_ratio": discordant_odds_ratio(b, c) if discordant else float("nan"),
        "proportion_a_better_discordant": prop_a,
    }


def mcnemar_comparison_row(
    table_a: PatientMethodTable,
    table_b: PatientMethodTable,
) -> Dict[str, Any]:
    """McNemar statistics for one method pair."""
    aligned = align_method_pair(table_a, table_b)
    y_true = aligned["y_true"]
    y_pred_a = aligned["y_pred_a"].astype(int)
    y_pred_b = aligned["y_pred_b"].astype(int)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    both_correct = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    a_correct_b_wrong = int((correct_a & ~correct_b).sum())
    a_wrong_b_correct = int((~correct_a & correct_b).sum())
    discordant_total = a_correct_b_wrong + a_wrong_b_correct

    test_type, statistic, p_value = mcnemar_test(a_correct_b_wrong, a_wrong_b_correct)
    effects = paired_comparison_effect_sizes(
        n_common=int(len(aligned)),
        a_correct_b_wrong=a_correct_b_wrong,
        a_wrong_b_correct=a_wrong_b_correct,
        both_correct=both_correct,
    )

    return {
        "method_a": table_a.method_name,
        "method_b": table_b.method_name,
        "n_common": int(len(aligned)),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": discordant_total,
        "accuracy_a": effects["accuracy_a"],
        "accuracy_b": effects["accuracy_b"],
        "accuracy_diff": effects["accuracy_diff"],
        "discordant_odds_ratio": effects["discordant_odds_ratio"],
        "proportion_a_better_discordant": effects["proportion_a_better_discordant"],
        "test_type": test_type,
        "statistic": statistic,
        "p_value": p_value,
    }


def run_mcnemar_comparisons(
    tables: Dict[str, PatientMethodTable],
    comparisons: Sequence[Tuple[str, str]] = MCNEMAR_COMPARISONS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method_a, method_b in comparisons:
        if method_a not in tables:
            LOGGER.warning("Skipping McNemar %s vs %s: %s missing", method_a, method_b, method_a)
            continue
        if method_b not in tables:
            LOGGER.warning("Skipping McNemar %s vs %s: %s missing", method_a, method_b, method_b)
            continue
        rows.append(mcnemar_comparison_row(tables[method_a], tables[method_b]))
    return pd.DataFrame(rows)


def sanity_check_reference_cohort(tables: Dict[str, PatientMethodTable]) -> List[str]:
    """Return warning messages for cohort size / manual class balance."""
    warnings: List[str] = []
    ref = None
    for candidate in ("v2_run_02", "v1", "cascade_standard"):
        if candidate in tables:
            ref = tables[candidate]
            break
    if ref is None and tables:
        ref = next(iter(tables.values()))
    if ref is None:
        warnings.append("No method tables loaded; cannot verify cohort size.")
        return warnings

    work = ref.frame
    valid = work["y_true"].notna() & work["y_true"].isin([0, 1])
    yt = work.loc[valid, "y_true"].astype(int)
    n = int(len(yt))
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())

    if n != EXPECTED_COMPLETE_PATIENTS:
        warnings.append(
            f"Expected {EXPECTED_COMPLETE_PATIENTS} complete patients, found {n} in {ref.method_name}."
        )
    if n_pos != EXPECTED_MANUAL_POSITIVES:
        warnings.append(
            f"Expected {EXPECTED_MANUAL_POSITIVES} manual positives, found {n_pos}."
        )
    if n_neg != EXPECTED_MANUAL_NEGATIVES:
        warnings.append(
            f"Expected {EXPECTED_MANUAL_NEGATIVES} manual negatives, found {n_neg}."
        )
    return warnings


def build_metrics_table(tables: Dict[str, PatientMethodTable]) -> pd.DataFrame:
    """Metrics + Wilson CIs for each loaded method."""
    method_order = [
        "icdsc",
        "icd10",
        "composite_or",
        "composite_and",
        "v1",
        "v2_run_02",
        "v2_run_03",
        "v2_run_04",
        "cascade_standard",
        "cascade_reviewer",
    ]
    rows: List[Dict[str, Any]] = []
    for name in method_order:
        if name not in tables:
            continue
        tbl = tables[name]
        rows.append(
            diagnostic_metrics_with_ci(
                tbl.frame["y_true"],
                tbl.frame["y_pred"],
                method_name=name,
            )
        )
    if not rows:
        return pd.DataFrame(columns=list(METRICS_OUTPUT_COLUMNS))
    out = pd.DataFrame(rows)
    return out[list(METRICS_OUTPUT_COLUMNS)]


def format_biostatistics_report(
    metrics: pd.DataFrame,
    mcnemar: pd.DataFrame,
    warnings: Sequence[str],
) -> str:
    lines = [
        "Biostatistical evaluation — manual validation cohort (patient-level)",
        "=" * 68,
        "",
        "Wilson 95% confidence intervals",
        "-" * 68,
        "Binomial proportions (sensitivity, specificity, PPV, NPV, accuracy) are",
        "summarised with Wilson score intervals (z = 1.96). Intervals are NA when",
        "the relevant denominator is zero.",
        "",
        "McNemar paired tests",
        "-" * 68,
        "Pairwise comparisons use patient-level correctness vs manual ground truth.",
        "Exact binomial McNemar is used when discordant pairs <= 25; otherwise",
        "chi-square McNemar with continuity correction.",
        "",
        "Effect sizes (McNemar comparisons)",
        "-" * 68,
        "accuracy_diff = paired accuracy(method_a) - accuracy(method_b) on n_common patients.",
        "discordant_odds_ratio = (b+0.5)/(c+0.5) where b = A correct/B wrong, c = A wrong/B correct.",
        "proportion_a_better_discordant = b/(b+c) among discordant patients only.",
        "Point estimates for sensitivity/specificity in diagnostic_metrics_with_ci.csv are",
        "already interpretable effect sizes vs manual ground truth (with Wilson CIs).",
        "",
        "Exploratory note",
        "-" * 68,
        "These analyses are exploratory: the validation cohort is small (n≈100),",
        "ground truth comes from a single manual review, and multiple comparisons",
        "were not multiplicity-adjusted.",
        "",
    ]

    if warnings:
        lines.append("Sanity-check warnings")
        lines.append("-" * 68)
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if not metrics.empty:
        lines.append("Diagnostic metrics summary")
        lines.append("-" * 68)
        for _, row in metrics.iterrows():
            lines.append(
                f"{row['method']}: n={int(row['n'])} "
                f"TP={int(row['TP'])} FP={int(row['FP'])} TN={int(row['TN'])} FN={int(row['FN'])} "
                f"sens={row['sensitivity']:.3f} [{row['sensitivity_ci_low']:.3f}, {row['sensitivity_ci_high']:.3f}] "
                f"spec={row['specificity']:.3f} [{row['specificity_ci_low']:.3f}, {row['specificity_ci_high']:.3f}] "
                f"acc={row['accuracy']:.3f} [{row['accuracy_ci_low']:.3f}, {row['accuracy_ci_high']:.3f}]"
            )
        lines.append("")

    if not mcnemar.empty:
        lines.append("McNemar comparisons")
        lines.append("-" * 68)
        for _, row in mcnemar.iterrows():
            p = row["p_value"]
            p_str = f"{p:.4f}" if pd.notna(p) else "NA"
            diff = row.get("accuracy_diff", float("nan"))
            diff_str = f"{diff:.3f}" if pd.notna(diff) else "NA"
            dor = row.get("discordant_odds_ratio", float("nan"))
            dor_str = f"{dor:.2f}" if pd.notna(dor) else "NA"
            lines.append(
                f"{row['method_a']} vs {row['method_b']}: "
                f"n={int(row['n_common'])} discordant={int(row['discordant_total'])} "
                f"({int(row['a_correct_b_wrong'])} vs {int(row['a_wrong_b_correct'])}) "
                f"acc_diff={diff_str} discordant_OR={dor_str} "
                f"test={row['test_type']} p={p_str}"
            )
        lines.append("")

    return "\n".join(lines)


def run_biostatistical_evaluation(
    output_dir: Path = BIOSTATISTICS_DIR,
    *,
    method_specs: Sequence[Tuple[str, Path, str]] = METHOD_FILE_SPECS,
    baseline_source: Path = BASELINE_SOURCE_PATH,
    comparisons: Sequence[Tuple[str, str]] = MCNEMAR_COMPARISONS,
) -> str:
    """Load patient-level outputs, compute CIs and McNemar tests, write reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_all_method_tables(method_specs=method_specs, baseline_source=baseline_source)
    warnings = sanity_check_reference_cohort(tables)
    for w in warnings:
        LOGGER.warning(w)

    metrics = build_metrics_table(tables)
    mcnemar = run_mcnemar_comparisons(tables, comparisons=comparisons)

    metrics_path = output_dir / "diagnostic_metrics_with_ci.csv"
    mcnemar_path = output_dir / "mcnemar_tests.csv"
    report_path = output_dir / "biostatistics_report.txt"

    metrics.to_csv(metrics_path, index=False)
    mcnemar.to_csv(mcnemar_path, index=False)
    report = format_biostatistics_report(metrics, mcnemar, warnings)
    report_path.write_text(report, encoding="utf-8")

    LOGGER.info("Wrote %s", metrics_path)
    LOGGER.info("Wrote %s", mcnemar_path)
    LOGGER.info("Wrote %s", report_path)
    return report


def main() -> None:
    report = run_biostatistical_evaluation()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
