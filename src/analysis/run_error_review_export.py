"""
LEGACY / DEPRECATED — multiclass ``manual_label_0_1_2`` review tied to baseline-as-truth.

PRIMARY manual validation: ``export_patient_validation_cohort`` + ``evaluate_manual_validation``.

Reads: outputs/comparisons/report_vs_baseline_comparison.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.pipeline.paths import MANUAL_REVIEW_DIR, REPORT_VS_BASELINE_PATH

LOGGER = logging.getLogger(__name__)

PRIMARY_BASELINES: Tuple[str, ...] = (
    "baseline_composite",
    "baseline_icd10",
    "baseline_icdsc_ge_4",
)

# Source columns in stable order (subset present in comparison CSV)
DATA_COLUMN_ORDER: Sequence[str] = (
    "PatientenID",
    "bericht",
    "klasse",
    "klassifikation",
    "signalstaerke",
    "anzahl_treffer",
    "delir_signale",
    "evidence_snippets",
    "kontext",
    "begruendung",
    "has_delir_icd10",
    "max_icdsc",
    "baseline_icd10",
    "baseline_icdsc_ge_1",
    "baseline_icdsc_ge_2",
    "baseline_icdsc_ge_3",
    "baseline_icdsc_ge_4",
    "baseline_icdsc_ge_5",
    "baseline_icdsc_0",
    "baseline_icdsc_1_to_3",
    "baseline_icdsc_ge_4_grouped",
    "llm_text_reduction_method",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_skipped_by_prefilter",
    "has_direct_delir_evidence",
    "has_indirect_delir_evidence",
    "has_negated_delir_evidence",
    "has_prophylaxis_or_risk_only",
    "has_alternative_explanation",
    "manual_review_candidate",
    "decision_rule_applied",
)

MANUAL_ANNOTATION_COLUMNS = (
    "manual_label_0_1_2",
    "manual_comment",
    "reviewer",
    "review_date",
)

ERROR_TYPES: Tuple[str, ...] = ("TP", "TN", "FP", "FN")
MAX_PER_CATEGORY = 5


def _bin(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _error_type_mask(df: pd.DataFrame, baseline_col: str) -> Dict[str, pd.Series]:
    pred = _bin(df["klasse"])
    base = _bin(df[baseline_col])
    return {
        "TP": (pred == 1) & (base == 1),
        "TN": (pred == 0) & (base == 0),
        "FP": (pred == 1) & (base == 0),
        "FN": (pred == 0) & (base == 1),
    }


def _pick_sample(df: pd.DataFrame, mask: pd.Series, k: int) -> pd.DataFrame:
    if not mask.any():
        return df.iloc[0:0].copy()
    sub = df.loc[mask].sort_values("PatientenID", key=lambda s: s.astype(str))
    return sub.head(k).copy()


def _build_review_frame(
    df: pd.DataFrame,
    baseline_name: str,
    error_type: str,
    sample: pd.DataFrame,
    review_seq_start: int,
) -> Tuple[pd.DataFrame, int]:
    if sample.empty:
        return sample, review_seq_start
    rows: List[Dict[str, object]] = []
    seq = review_seq_start
    for _, row in sample.iterrows():
        seq += 1
        entry: Dict[str, object] = {
            "review_id": f"mr_{baseline_name}_{error_type}_{seq:04d}",
            "baseline_name": baseline_name,
            "error_type": error_type,
            "baseline_value": int(_bin(pd.Series([row[baseline_name]])).iloc[0]),
        }
        for c in DATA_COLUMN_ORDER:
            if c in row.index:
                val = row[c]
                if pd.isna(val):
                    entry[c] = ""
                else:
                    entry[c] = val
            else:
                entry[c] = ""
        for c in MANUAL_ANNOTATION_COLUMNS:
            entry[c] = ""
        rows.append(entry)
    out = pd.DataFrame(rows)
    return out, seq


def _select_export_columns(df: pd.DataFrame) -> List[str]:
    present = [c for c in DATA_COLUMN_ORDER if c in df.columns]
    missing = [c for c in DATA_COLUMN_ORDER if c not in df.columns]
    for c in missing:
        LOGGER.warning("Comparison CSV missing optional column %s", c)
    if "evidence_snippets" not in present:
        LOGGER.warning("Column evidence_snippets missing; export will leave cells empty.")
    return present


def run_manual_review_export(
    cmp_path: Path = REPORT_VS_BASELINE_PATH,
    out_dir: Path = MANUAL_REVIEW_DIR,
    max_per_category: int = MAX_PER_CATEGORY,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cmp_path.exists():
        raise FileNotFoundError(f"Comparison file missing: {cmp_path}. Run compare_reports_vs_baseline first.")

    df = pd.read_csv(cmp_path)
    if "PatientenID" not in df.columns or "klasse" not in df.columns:
        raise ValueError(f"Expected PatientenID and klasse columns. Found: {list(df.columns)}")

    df = df.copy()
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    data_cols = _select_export_columns(df)

    summary_rows: List[Dict[str, object]] = []
    all_parts: List[pd.DataFrame] = []
    global_seq = 0

    for baseline_name in PRIMARY_BASELINES:
        if baseline_name not in df.columns:
            LOGGER.warning("Skipping missing baseline column: %s", baseline_name)
            continue

        masks = _error_type_mask(df, baseline_name)
        n_eval = len(df)
        cm = {et: int(masks[et].sum()) for et in ERROR_TYPES}

        per_cat_frames: List[pd.DataFrame] = []
        exported_counts: Dict[str, int] = {}

        for et in ERROR_TYPES:
            sample = _pick_sample(df, masks[et], max_per_category)
            exported_counts[et] = len(sample)
            frame, global_seq = _build_review_frame(df, baseline_name, et, sample, global_seq)
            if not frame.empty:
                per_cat_frames.append(frame)

        if per_cat_frames:
            baseline_df = pd.concat(per_cat_frames, ignore_index=True)
            safe = baseline_name.replace("/", "_")
            baseline_df.to_csv(out_dir / f"manual_review_cases_{safe}.csv", index=False)
            all_parts.append(baseline_df)
        else:
            safe = baseline_name.replace("/", "_")
            empty_cols = (
                ["review_id", "baseline_name", "error_type", "baseline_value"]
                + list(data_cols)
                + list(MANUAL_ANNOTATION_COLUMNS)
            )
            pd.DataFrame(columns=empty_cols).to_csv(out_dir / f"manual_review_cases_{safe}.csv", index=False)

        summary_rows.append(
            {
                "baseline_name": baseline_name,
                "total_evaluable_rows": n_eval,
                "count_TP": cm["TP"],
                "count_TN": cm["TN"],
                "count_FP": cm["FP"],
                "count_FN": cm["FN"],
                "exported_TP": exported_counts["TP"],
                "exported_TN": exported_counts["TN"],
                "exported_FP": exported_counts["FP"],
                "exported_FN": exported_counts["FN"],
            }
        )

    if all_parts:
        pd.concat(all_parts, ignore_index=True).to_csv(out_dir / "manual_review_cases_all.csv", index=False)
    else:
        empty_cols = (
            ["review_id", "baseline_name", "error_type", "baseline_value"]
            + list(data_cols)
            + list(MANUAL_ANNOTATION_COLUMNS)
        )
        pd.DataFrame(columns=empty_cols).to_csv(out_dir / "manual_review_cases_all.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "baseline_name",
                "total_evaluable_rows",
                "count_TP",
                "count_TN",
                "count_FP",
                "count_FN",
                "exported_TP",
                "exported_TN",
                "exported_FP",
                "exported_FN",
            ]
        )
    summary_df.to_csv(out_dir / "manual_review_summary.csv", index=False)

    report_lines = [
        "Manual review export (validation / scientific QC)",
        "",
        f"Source: {cmp_path}",
        f"Output directory: {out_dir}",
        "",
        "Primary baselines:",
        "  - baseline_icd10",
        "  - baseline_icdsc_ge_4",
        "",
        "Confusion cell definitions (klasse and baseline clipped to 0/1):",
        "  TP: klasse == 1 and baseline == 1",
        "  TN: klasse == 0 and baseline == 0",
        "  FP: klasse == 1 and baseline == 0",
        "  FN: klasse == 0 and baseline == 1",
        "",
        "Per baseline:",
    ]

    for _, srow in summary_df.iterrows():
        if srow.get("baseline_name") is None or (isinstance(srow.get("baseline_name"), float) and pd.isna(srow.get("baseline_name"))):
            continue
        report_lines.extend(
            [
                f"  Baseline: {srow['baseline_name']}",
                f"    Total evaluable rows: {int(srow['total_evaluable_rows'])}",
                f"    Counts — TP: {int(srow['count_TP'])}, TN: {int(srow['count_TN'])}, "
                f"FP: {int(srow['count_FP'])}, FN: {int(srow['count_FN'])}",
                f"    Exported for manual review (up to {max_per_category} per category) — "
                f"TP: {int(srow['exported_TP'])}, TN: {int(srow['exported_TN'])}, "
                f"FP: {int(srow['exported_FP'])}, FN: {int(srow['exported_FN'])}",
                "",
            ]
        )

    report_lines.extend(
        [
            "Interpretation:",
            "  False positives may represent model overcalling OR structured baseline undercoding.",
            "  False negatives may represent missed text evidence OR baseline-only signal.",
            "  Manual review is required before changing prompts or interpreting performance.",
            "",
            "Manual annotation columns (empty until reviewed):",
            "  manual_label_0_1_2 — 0 = no delirium evidence, 1 = possible/indirect, 2 = clear documented delirium, empty = not reviewed",
            "  manual_comment, reviewer, review_date",
            "",
            f"Rows in comparison file: {len(df)}",
        ]
    )

    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Manual review summary: {out_dir / 'manual_review_summary.csv'}")
    print(f"Report: {out_dir / 'report.txt'}")


def run_error_review(*args, **kwargs) -> None:
    """Backward-compatible name for analysis suites."""
    run_manual_review_export(*args, **kwargs)


def main() -> None:
    run_manual_review_export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
