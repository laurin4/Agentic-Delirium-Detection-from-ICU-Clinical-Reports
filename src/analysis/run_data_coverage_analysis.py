"""
Pre-model data coverage: Berichte.csv vs structured_baseline.csv.

Does not call the LLM pipeline. Raises FileNotFoundError if inputs are missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.cohort_counts import (
    PLOT_TITLE_SUFFIX,
    baseline_patient_id_set,
    berichte_patient_id_set,
    current_cohort_counts_dataframe,
    load_and_compute_current_cohort_counts,
    print_current_cohort_counts,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    DATA_COVERAGE_ANALYSIS_DIR,
    DATA_COVERAGE_PLOTS_DIR,
    DATA_COVERAGE_TABLES_DIR,
    ICD10_PATH,
    ICDSC_PATH,
    STRUCTURED_BASELINE_PATH,
)

LOGGER = logging.getLogger(__name__)
PLOT_TITLE_SIZE = 15
PLOT_SUBTITLE_SIZE = 11
PLOT_LABEL_SIZE = 12
PLOT_TICK_SIZE = 11
PLOT_ANNOTATION_SIZE = 11


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mpl_config_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _clear_stale_plots(plot_dir: Path) -> None:
    """Remove old PNGs so outdated cohort figures cannot be mistaken for current runs."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    for png in plot_dir.glob("*.png"):
        png.unlink()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def _aggregate_baseline_flags(baseline_df: pd.DataFrame, pid: str) -> Dict[str, Any]:
    """Per-PatientenID clinical flags (baseline is patient-level after load dedupe)."""
    sub = baseline_df[baseline_df["PatientenID"] == pid]
    out: Dict[str, Any] = {"has_delir_icd10_max": None, "max_icdsc_max": None}
    if sub.empty:
        return out
    if "has_delir_icd10" in sub.columns:
        out["has_delir_icd10_max"] = float(pd.to_numeric(sub["has_delir_icd10"], errors="coerce").fillna(0).max())
    if "max_icdsc" in sub.columns:
        out["max_icdsc_max"] = float(pd.to_numeric(sub["max_icdsc"], errors="coerce").fillna(0).max())
    return out


def _duplicate_distribution_and_summary(
    df: pd.DataFrame, id_col: str, dataset_label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        summary = pd.DataFrame(
            [
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "total_rows",
                    "value": 0,
                },
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "unique_ids",
                    "value": 0,
                },
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "max_rows_per_id",
                    "value": 0,
                },
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "mean_rows_per_id",
                    "value": 0.0,
                },
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "median_rows_per_id",
                    "value": 0.0,
                },
                {
                    "dataset": dataset_label,
                    "category": "summary",
                    "key": "patients_with_multiple_rows",
                    "value": 0,
                },
            ]
        )
        return summary, pd.DataFrame(columns=["dataset", "category", "key", "value"])

    vc = df.groupby(id_col, dropna=False).size()
    dist = vc.value_counts().sort_index()

    summary_rows = [
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "total_rows",
            "value": int(len(df)),
        },
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "unique_ids",
            "value": int(len(vc)),
        },
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "max_rows_per_id",
            "value": int(vc.max()),
        },
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "mean_rows_per_id",
            "value": round(float(vc.mean()), 6),
        },
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "median_rows_per_id",
            "value": round(float(vc.median()), 6),
        },
        {
            "dataset": dataset_label,
            "category": "summary",
            "key": "patients_with_multiple_rows",
            "value": int((vc > 1).sum()),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    dist_rows = [
        {
            "dataset": dataset_label,
            "category": "distribution",
            "key": f"rows_per_id_{int(k)}",
            "value": int(v),
        }
        for k, v in dist.items()
    ]
    dist_df = pd.DataFrame(dist_rows)
    return summary_df, dist_df


def _load_raw_with_patient_id(path: Path, label: str) -> pd.DataFrame:
    _require_file(path, label)
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if "PatientID" in df.columns:
        pid_col = "PatientID"
    elif "PatientenID" in df.columns:
        pid_col = "PatientenID"
    else:
        raise ValueError(f"{label} must contain 'PatientID' or 'PatientenID'. Found: {list(df.columns)}")
    out = df.copy()
    out["PatientID"] = _normalize_id_series(out[pid_col])
    out = out[out["PatientID"].str.len() > 0]
    out = out[out["PatientID"].str.lower() != "nan"]
    return out


def _plot_raw_source_sizes(sizes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    x = np.arange(len(sizes))
    w = 0.36
    rows_bars = ax.bar(x - w / 2, sizes["n_rows"], width=w, label="Rows", color="#1d4ed8")
    uniq_bars = ax.bar(x + w / 2, sizes["n_unique_patient_ids"], width=w, label="Unique PatientIDs", color="#60a5fa")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes["dataset"], rotation=12, ha="right", fontsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    ax.set_ylabel("Count", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Raw source sizes: rows vs unique PatientIDs" + PLOT_TITLE_SUFFIX,
        fontsize=PLOT_TITLE_SIZE,
        pad=14,
    )
    for bars in (rows_bars, uniq_bars):
        for bar in bars:
            h = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{h}",
                ha="center",
                va="bottom",
                fontsize=PLOT_ANNOTATION_SIZE,
            )
    ax.legend(fontsize=PLOT_LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_patient_level_cohort_sizes(
    n_berichte_unique: int,
    n_baseline_unique: int,
    n_overlap: int,
    out_path: Path,
) -> None:
    labels = ["Berichte unique PatientIDs", "structured_baseline unique PatientIDs", "Berichte ∩ structured_baseline"]
    vals = [n_berichte_unique, n_baseline_unique, n_overlap]
    colors = ["#1d4ed8", "#64748b", "#16a34a"]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Unique patient IDs", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Patient-level cohort sizes (Berichte vs structured_baseline)" + PLOT_TITLE_SUFFIX,
        fontsize=PLOT_TITLE_SIZE,
        pad=14,
    )
    ax.tick_params(axis="x", rotation=10, labelsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_SIZE,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overlap_distribution(
    n_berichte_only: int,
    n_overlap: int,
    n_baseline_only: int,
    out_path: Path,
) -> None:
    labels = ["Berichte only", "Overlap", "Baseline only"]
    vals = [n_berichte_only, n_overlap, n_baseline_only]
    colors = ["#ea580c", "#16a34a", "#64748b"]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Unique patient IDs", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Overlap distribution: Berichte vs structured_baseline" + PLOT_TITLE_SUFFIX,
        fontsize=PLOT_TITLE_SIZE,
        pad=14,
    )
    ax.tick_params(axis="x", rotation=12, labelsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_SIZE,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_unmatched_counts(
    n_berichte_without_baseline: int,
    n_baseline_without_berichte: int,
    n_overlap: int,
    out_path: Path,
) -> None:
    labels = [
        "Berichte without baseline",
        "Baseline without Berichte",
        "Overlap (matched)",
    ]
    vals = [n_berichte_without_baseline, n_baseline_without_berichte, n_overlap]
    colors = ["#ea580c", "#64748b", "#16a34a"]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Unique patient IDs", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Unmatched vs matched patient IDs (Berichte vs structured_baseline)" + PLOT_TITLE_SUFFIX,
        fontsize=PLOT_TITLE_SIZE,
        pad=14,
    )
    ax.tick_params(axis="x", rotation=14, labelsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_SIZE,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_duplicates_histogram(
    berichte_vc: pd.Series,
    baseline_vc: pd.Series,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)

    def _panel(ax, vc: pd.Series, title: str) -> None:
        if vc.empty:
            ax.text(0.5, 0.5, "No rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        idx = vc.index.astype(int)
        ax.bar(idx.astype(str), vc.values.astype(int), color="#7c3aed")
        ax.set_xlabel("Rows per patient ID")
        ax.set_ylabel("Number of patient IDs")
        ax.set_title(title)

    _panel(axes[0], berichte_vc, "Berichte.csv")
    _panel(axes[1], baseline_vc, "structured_baseline.csv")
    fig.suptitle("Distribution: rows per patient ID", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_berichte_matching_pie(
    n_berichte_matched: int,
    n_berichte_unmatched: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    vals = [n_berichte_matched, n_berichte_unmatched]
    labels = ["Matched Berichte patients", "Unmatched Berichte patients"]
    colors = ["#16a34a", "#ea580c"]
    total = max(sum(vals), 1)

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    wedges, _texts, _autotexts = ax.pie(
        vals,
        labels=None,
        colors=colors,
        autopct=_autopct,
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        textprops={"fontsize": PLOT_ANNOTATION_SIZE},
    )
    ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=PLOT_LABEL_SIZE,
    )
    ax.set_title(
        "Berichte PatientID matching against structured baseline",
        fontsize=PLOT_TITLE_SIZE,
        pad=24,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_berichte_matching_bar(
    n_berichte_matched: int,
    n_berichte_unmatched: int,
    n_berichte_unique: int,
    out_path: Path,
) -> None:
    labels = ["Matched Berichte patients", "Unmatched Berichte patients"]
    vals = [n_berichte_matched, n_berichte_unmatched]
    colors = ["#16a34a", "#ea580c"]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Number of Berichte patients", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Berichte coverage: matched vs unmatched PatientIDs\n"
        f"{n_berichte_unique} Berichte patients evaluated",
        fontsize=PLOT_TITLE_SIZE,
        pad=16,
    )
    ax.tick_params(axis="x", rotation=8, labelsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    for bar, val in zip(bars, vals):
        pct = _pct(int(val), n_berichte_unique) * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(val)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_SIZE,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_cohort_size_context(
    n_berichte_unique: int,
    n_baseline_unique: int,
    n_overlap: int,
    out_path: Path,
) -> None:
    labels = ["Berichte unique patients", "Structured baseline unique patients", "Overlap"]
    vals = [n_berichte_unique, n_baseline_unique, n_overlap]
    colors = ["#2563eb", "#64748b", "#16a34a"]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Unique patient IDs (log scale)", fontsize=PLOT_LABEL_SIZE)
    ax.set_title(
        "Cohort size context: Berichte vs structured_baseline" + PLOT_TITLE_SUFFIX,
        fontsize=PLOT_TITLE_SIZE,
        pad=16,
    )
    ax.tick_params(axis="x", rotation=10, labelsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
    for bar, val in zip(bars, vals):
        y = max(float(val) * 1.08, float(val) + 1.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_SIZE,
        )
    footer = (
        f"Overlap coverage: {n_overlap} / {n_berichte_unique} Berichte patients "
        f"({_pct(n_overlap, n_berichte_unique) * 100.0:.1f}%)"
    )
    fig.text(0.5, 0.03, footer, ha="center", va="center", fontsize=PLOT_ANNOTATION_SIZE)
    fig.tight_layout()
    plt.subplots_adjust(top=0.84, bottom=0.17)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overlap_venn_style(
    n_berichte_only: int,
    n_overlap: int,
    n_baseline_only: int,
    n_berichte_unique: int,
    n_baseline_unique: int,
    out_path: Path,
) -> None:
    vals = [n_berichte_only, n_overlap, n_baseline_only]
    labels = ["Berichte only", "Overlap", "Baseline only"]
    colors = ["#ea580c", "#16a34a", "#64748b"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    left = 0
    for val, label, color in zip(vals, labels, colors):
        ax.barh(["PatientID universe"], [val], left=left, color=color, height=0.55, label=label)
        ax.text(
            left + val / 2.0,
            0,
            f"{label}\n{int(val)}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )
        left += val
    pct_of_berichte = _pct(n_overlap, n_berichte_unique) * 100.0
    pct_of_baseline = _pct(n_overlap, n_baseline_unique) * 100.0
    ax.set_title("PatientID overlap between Berichte and structured baseline")
    ax.set_xlabel("Unique patient ID count")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))
    ax.text(
        0.01,
        -0.28,
        f"Overlap as % of Berichte: {pct_of_berichte:.1f}% | Overlap as % of baseline: {pct_of_baseline:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))

    berichte, baseline, cohort_counts, berichte_path, baseline_path = load_and_compute_current_cohort_counts(
        BERICHTE_INPUT_PATH,
        STRUCTURED_BASELINE_PATH,
    )
    print_current_cohort_counts(cohort_counts, berichte_path=berichte_path, baseline_path=baseline_path)

    raw_icd = _load_raw_with_patient_id(ICD10_PATH, "ICD.csv")
    raw_icdsc = _load_raw_with_patient_id(ICDSC_PATH, "ICDSC.csv")

    DATA_COVERAGE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_COVERAGE_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_COVERAGE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_stale_plots(DATA_COVERAGE_PLOTS_DIR)

    current_cohort_counts_dataframe(cohort_counts).to_csv(
        DATA_COVERAGE_TABLES_DIR / "current_cohort_counts.csv",
        index=False,
    )

    b_ids = berichte_patient_id_set(berichte)
    m_ids = baseline_patient_id_set(baseline)

    inter = b_ids & m_ids
    ber_only = b_ids - m_ids
    base_only = m_ids - b_ids
    union_ids = b_ids | m_ids
    n_berichte_unique = cohort_counts["berichte_unique_patientids"]
    n_baseline_unique = cohort_counts["structured_baseline_unique_patientids"]
    n_overlap = cohort_counts["overlap_patientids"]
    n_berichte_unmatched = cohort_counts["berichte_without_baseline"]
    n_baseline_without_berichte = cohort_counts["baseline_without_berichte"]

    berichte_matching_summary = pd.DataFrame(
        [
            {
                "n_berichte_unique": n_berichte_unique,
                "n_baseline_unique": n_baseline_unique,
                "n_overlap": n_overlap,
                "n_berichte_unmatched": n_berichte_unmatched,
                "n_baseline_without_berichte": n_baseline_without_berichte,
                "percent_berichte_matched": _pct(n_overlap, n_berichte_unique),
                "percent_berichte_unmatched": _pct(n_berichte_unmatched, n_berichte_unique),
                "percent_baseline_with_berichte": _pct(n_overlap, n_baseline_unique),
            }
        ]
    )
    berichte_matching_summary.to_csv(
        DATA_COVERAGE_TABLES_DIR / "berichte_matching_summary.csv",
        index=False,
    )
    pd.DataFrame({"PatientID": sorted(ber_only)}).to_csv(
        DATA_COVERAGE_TABLES_DIR / "berichte_unmatched_patient_ids.csv",
        index=False,
    )
    pd.DataFrame({"PatientenID": sorted(base_only)}).to_csv(
        DATA_COVERAGE_TABLES_DIR / "baseline_without_berichte_patient_ids.csv",
        index=False,
    )

    raw_sizes_df = pd.DataFrame(
        [
            {
                "dataset": "Berichte.csv",
                "n_rows": len(berichte),
                "n_unique_patient_ids": len(b_ids),
            },
            {
                "dataset": "ICD.csv",
                "n_rows": len(raw_icd),
                "n_unique_patient_ids": int(raw_icd["PatientID"].nunique()),
            },
            {
                "dataset": "ICDSC.csv",
                "n_rows": len(raw_icdsc),
                "n_unique_patient_ids": int(raw_icdsc["PatientID"].nunique()),
            },
        ]
    )
    raw_sizes_df.to_csv(DATA_COVERAGE_TABLES_DIR / "raw_source_sizes.csv", index=False)

    patient_level_sizes_df = pd.DataFrame(
        [
            {"cohort": "Berichte_unique_patient_ids", "n_unique_patient_ids": n_berichte_unique},
            {"cohort": "structured_baseline_unique_patient_ids", "n_unique_patient_ids": n_baseline_unique},
            {"cohort": "berichte_intersection_structured_baseline", "n_unique_patient_ids": n_overlap},
        ]
    )
    patient_level_sizes_df.to_csv(DATA_COVERAGE_TABLES_DIR / "patient_level_cohort_sizes.csv", index=False)

    overlap_rows: List[Dict[str, object]] = [
        {"category": "berichte_intersect_baseline", "n_unique_patient_ids": len(inter)},
        {"category": "berichte_only", "n_unique_patient_ids": len(ber_only)},
        {"category": "baseline_only", "n_unique_patient_ids": len(base_only)},
        {"category": "union_unique_patient_ids", "n_unique_patient_ids": len(union_ids)},
    ]

    if inter:
        sub = baseline[baseline["PatientenID"].isin(inter)].copy()
        if "has_delir_icd10" in sub.columns:
            h = pd.to_numeric(sub["has_delir_icd10"], errors="coerce").fillna(0).astype(int)
            overlap_rows.append(
                {
                    "category": "in_overlap_with_icd10_delir_flag_1",
                    "n_unique_patient_ids": int((h == 1).sum()),
                }
            )
        if "max_icdsc" in sub.columns:
            mx = pd.to_numeric(sub["max_icdsc"], errors="coerce")
            overlap_rows.append(
                {
                    "category": "in_overlap_with_icdsc_max_score_gt_0",
                    "n_unique_patient_ids": int((mx.fillna(0) > 0).sum()),
                }
            )

    pd.DataFrame(overlap_rows).to_csv(DATA_COVERAGE_TABLES_DIR / "overlap_counts.csv", index=False)

    n_berichte_with_icd10_delir = 0
    n_berichte_without_icd10_delir = 0
    n_berichte_with_icdsc_signal = 0
    n_berichte_without_icdsc_signal = 0
    for pid in b_ids:
        if pid not in m_ids:
            n_berichte_without_icd10_delir += 1
            n_berichte_without_icdsc_signal += 1
            continue
        agg = _aggregate_baseline_flags(baseline, pid)
        hmax = agg["has_delir_icd10_max"]
        if hmax is not None and int(hmax) == 1:
            n_berichte_with_icd10_delir += 1
        else:
            n_berichte_without_icd10_delir += 1
        mx = agg["max_icdsc_max"]
        if mx is not None and mx > 0:
            n_berichte_with_icdsc_signal += 1
        else:
            n_berichte_without_icdsc_signal += 1

    unmatched_rows = [
        {
            "metric_group": "linkage",
            "metric": "berichte_without_baseline_patientid",
            "n_unique_patient_ids": len(ber_only),
            "definition": "PatientID in Berichte.csv absent from structured_baseline.csv.",
        },
        {
            "metric_group": "linkage",
            "metric": "baseline_without_berichte_patientid",
            "n_unique_patient_ids": len(base_only),
            "definition": "Baseline patients without report in Berichte subset.",
        },
        {
            "metric_group": "linkage",
            "metric": "berichte_with_baseline_patientid",
            "n_unique_patient_ids": len(inter),
            "definition": "PatientID present in both Berichte.csv and structured_baseline.csv.",
        },
        {
            "metric_group": "linkage",
            "metric": "all_patientid_overlap",
            "n_unique_patient_ids": len(inter),
            "definition": "Alias of intersection size (same as berichte_with_baseline_patientid).",
        },
        {
            "metric_group": "clinical_positivity",
            "metric": "berichte_patients_with_icd10_delir",
            "n_unique_patient_ids": n_berichte_with_icd10_delir,
            "definition": "Berichte patient IDs with has_delir_icd10==1 in baseline (after duplicate-ID aggregation).",
        },
        {
            "metric_group": "clinical_positivity",
            "metric": "berichte_patients_without_icd10_delir",
            "n_unique_patient_ids": n_berichte_without_icd10_delir,
            "definition": "Berichte patient IDs without has_delir_icd10==1 (includes IDs not linked to baseline).",
        },
        {
            "metric_group": "clinical_positivity",
            "metric": "berichte_patients_with_icdsc_signal",
            "n_unique_patient_ids": n_berichte_with_icdsc_signal,
            "definition": "Berichte patient IDs with max_icdsc>0 in structured_baseline.",
        },
        {
            "metric_group": "clinical_positivity",
            "metric": "berichte_patients_without_icdsc_signal",
            "n_unique_patient_ids": n_berichte_without_icdsc_signal,
            "definition": "Berichte patient IDs without ICDSC signal (includes IDs not linked to baseline).",
        },
    ]

    pd.DataFrame(unmatched_rows).to_csv(DATA_COVERAGE_TABLES_DIR / "unmatched_counts.csv", index=False)

    s1, d1 = _duplicate_distribution_and_summary(berichte, "PatientID", "Berichte.csv")
    s2, d2 = _duplicate_distribution_and_summary(baseline, "PatientenID", "structured_baseline.csv")
    duplicates_df = pd.concat([s1, d1, s2, d2], ignore_index=True)
    duplicates_df.to_csv(DATA_COVERAGE_TABLES_DIR / "duplicates_summary.csv", index=False)

    _plot_raw_source_sizes(raw_sizes_df, DATA_COVERAGE_PLOTS_DIR / "raw_source_sizes.png")
    _plot_patient_level_cohort_sizes(
        n_berichte_unique=n_berichte_unique,
        n_baseline_unique=n_baseline_unique,
        n_overlap=n_overlap,
        out_path=DATA_COVERAGE_PLOTS_DIR / "patient_level_cohort_sizes.png",
    )
    _plot_overlap_distribution(
        n_berichte_only=len(ber_only),
        n_overlap=n_overlap,
        n_baseline_only=len(base_only),
        out_path=DATA_COVERAGE_PLOTS_DIR / "overlap_distribution.png",
    )
    _plot_unmatched_counts(
        n_berichte_without_baseline=n_berichte_unmatched,
        n_baseline_without_berichte=n_baseline_without_berichte,
        n_overlap=n_overlap,
        out_path=DATA_COVERAGE_PLOTS_DIR / "unmatched_counts.png",
    )
    _plot_berichte_matching_pie(
        n_berichte_matched=n_overlap,
        n_berichte_unmatched=n_berichte_unmatched,
        out_path=DATA_COVERAGE_PLOTS_DIR / "berichte_matching_pie.png",
    )
    _plot_berichte_matching_bar(
        n_berichte_matched=n_overlap,
        n_berichte_unmatched=n_berichte_unmatched,
        n_berichte_unique=n_berichte_unique,
        out_path=DATA_COVERAGE_PLOTS_DIR / "berichte_matching_bar.png",
    )
    _plot_cohort_size_context(
        n_berichte_unique=n_berichte_unique,
        n_baseline_unique=n_baseline_unique,
        n_overlap=n_overlap,
        out_path=DATA_COVERAGE_PLOTS_DIR / "cohort_size_context.png",
    )

    report_lines = [
        "Data coverage analysis (pre-model, current final data)",
        "  Dokumentationsblatt (bertyp) rows excluded from Berichte counts; raw CSV unchanged.",
        "",
        f"Berichte path: {berichte_path.resolve()}",
        f"Baseline path: {baseline_path.resolve()}",
        "",
        "Current cohort counts (patient-level unique IDs)",
    ]
    for name, val in cohort_counts.items():
        report_lines.append(f"  {name}: {val}")
    report_lines.extend(
        [
            "",
            "Raw data row counts and unique patient IDs",
        ]
    )
    report_lines.extend(
        [
        f"  Berichte.csv -> rows: {len(berichte)}, unique PatientID: {len(b_ids)}",
        f"  ICD.csv -> rows: {len(raw_icd)}, unique PatientID: {int(raw_icd['PatientID'].nunique())}",
        f"  ICDSC.csv -> rows: {len(raw_icdsc)}, unique PatientID: {int(raw_icdsc['PatientID'].nunique())}",
        "",
        "Patient-level cohort sizes",
        f"  Berichte unique PatientIDs: {n_berichte_unique}",
        f"  structured_baseline unique PatientIDs: {n_baseline_unique}",
        f"  Berichte ∩ structured_baseline: {n_overlap}",
        "",
        "Aggregation note",
        "  structured_baseline.csv is built from ICD.csv + ICDSC.csv (patient-level).",
        "  Duplicate PatientenID rows in the file are deduplicated before cohort counts.",
        "",
        "Overlap (unique patient IDs)",
        f"  Intersection: {len(inter)}",
        f"  Berichte only: {len(ber_only)}",
        f"  Baseline only: {len(base_only)}",
        f"  Union: {len(union_ids)}",
        "",
        "PatientID linkage (coverage only, no clinical positivity)",
        f"  berichte_without_baseline_patientid: {len(ber_only)}",
        f"  baseline_without_berichte_patientid: {len(base_only)}",
        f"  berichte_with_baseline_patientid: {len(inter)}",
        f"  all_patientid_overlap: {len(inter)}",
        "",
        "Berichte-centric matching interpretation",
        f"  Berichte unique patients: {n_berichte_unique}",
        f"  Structured baseline unique patients: {n_baseline_unique}",
        f"  Matched Berichte patients: {n_overlap} ({_pct(n_overlap, n_berichte_unique) * 100.0:.1f}%)",
        f"  Unmatched Berichte patients: {n_berichte_unmatched} ({_pct(n_berichte_unmatched, n_berichte_unique) * 100.0:.1f}%)",
        f"  Baseline patients without Berichte: {n_baseline_without_berichte}",
        "  Interpretation: Compare unique patient IDs only (not raw ICD/ICDSC row counts).",
        "  First 30 unmatched Berichte PatientIDs:",
        ]
    )
    for pid in sorted(list(ber_only))[:30]:
        report_lines.append(f"    - {pid}")
    report_lines.extend(
        [
            "  Note: unmatched Berichte PatientIDs should be checked with the data provider.",
            "",
            "Recommended primary figures",
            "  - patient_level_cohort_sizes.png",
            "  - cohort_size_context.png",
            "  - overlap_distribution.png",
            "  - unmatched_counts.png",
            "  - raw_source_sizes.png",
            "",
            "Interpretation notes",
            "  Cohort sizes use outputs/baseline/structured_baseline.csv (not legacy diagnosis lists).",
            "  Clinically relevant linkage metric: overlap_patientids / berichte_unique_patientids.",
            "",
            "Clinical positivity among Berichte patient IDs",
        ]
    )
    for r in unmatched_rows:
        if r["metric_group"] == "clinical_positivity":
            report_lines.append(f"  {r['metric']}: {r['n_unique_patient_ids']}")

    dup_b = duplicates_df[(duplicates_df["dataset"] == "Berichte.csv") & (duplicates_df["category"] == "summary")]
    dup_m = duplicates_df[(duplicates_df["dataset"] == "structured_baseline.csv") & (duplicates_df["category"] == "summary")]
    report_lines.extend(
        [
            "",
            "Duplicate rows (per patient ID)",
            "  Berichte.csv:",
        ]
    )
    for _, row in dup_b.iterrows():
        report_lines.append(f"    {row['key']}: {row['value']}")
    report_lines.append("  structured_baseline.csv:")
    for _, row in dup_m.iterrows():
        report_lines.append(f"    {row['key']}: {row['value']}")

    report_lines.extend(
        [
            "",
            f"Tables: {DATA_COVERAGE_TABLES_DIR}",
            f"Plots: {DATA_COVERAGE_PLOTS_DIR}",
        ]
    )
    (DATA_COVERAGE_ANALYSIS_DIR / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote current_cohort_counts: {DATA_COVERAGE_TABLES_DIR / 'current_cohort_counts.csv'}")
    print(f"Wrote tables under: {DATA_COVERAGE_TABLES_DIR}")
    print(f"Wrote plots under: {DATA_COVERAGE_PLOTS_DIR}")
    print(f"Wrote report: {DATA_COVERAGE_ANALYSIS_DIR / 'report.txt'}")
    print_current_cohort_counts(cohort_counts, berichte_path=berichte_path, baseline_path=baseline_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
