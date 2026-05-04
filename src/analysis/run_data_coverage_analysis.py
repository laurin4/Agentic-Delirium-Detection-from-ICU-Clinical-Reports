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

from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    DATA_COVERAGE_ANALYSIS_DIR,
    DATA_COVERAGE_PLOTS_DIR,
    DATA_COVERAGE_TABLES_DIR,
    STRUCTURED_BASELINE_PATH,
)

LOGGER = logging.getLogger(__name__)


def _mpl_config_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def _load_berichte(path: Path) -> pd.DataFrame:
    _require_file(path, "Berichte.csv (primary text dataset)")
    last_err: Optional[BaseException] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", dtype=str, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc
            LOGGER.warning("Berichte read failed with encoding %s: %s", enc, exc)
            continue
    else:
        raise ValueError(f"Berichte.csv could not be read: {path}") from last_err

    df.columns = [str(c).strip() for c in df.columns]
    if "PatientID" not in df.columns:
        raise ValueError(f"Berichte.csv must contain 'PatientID'. Found: {list(df.columns)}")
    df = df.copy()
    df["PatientID"] = _normalize_id_series(df["PatientID"])
    df = df[df["PatientID"].str.len() > 0]
    df = df[df["PatientID"].str.lower() != "nan"]
    return df


def _load_baseline(path: Path) -> pd.DataFrame:
    _require_file(path, "structured_baseline.csv")
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if "PatientenID" not in df.columns:
        raise ValueError(f"structured_baseline.csv must contain 'PatientenID'. Found: {list(df.columns)}")
    df = df.copy()
    df["PatientenID"] = _normalize_id_series(df["PatientenID"])
    df = df[df["PatientenID"].str.len() > 0]
    df = df[df["PatientenID"].str.lower() != "nan"]
    return df


def _id_set_berichte(df: pd.DataFrame) -> Set[str]:
    return set(df["PatientID"].unique())


def _id_set_baseline(df: pd.DataFrame) -> Set[str]:
    return set(df["PatientenID"].unique())


def _aggregate_baseline_flags(baseline_df: pd.DataFrame, pid: str) -> Dict[str, Any]:
    """Per-PatientenID aggregates across duplicate baseline rows."""
    sub = baseline_df[baseline_df["PatientenID"] == pid]
    out: Dict[str, Any] = {"has_delir_icd10_max": None, "max_icdsc_max": None, "n_icdsc_measurements_sum": 0.0}
    if sub.empty:
        return out
    if "has_delir_icd10" in sub.columns:
        out["has_delir_icd10_max"] = float(pd.to_numeric(sub["has_delir_icd10"], errors="coerce").fillna(0).max())
    if "max_icdsc" in sub.columns:
        out["max_icdsc_max"] = float(pd.to_numeric(sub["max_icdsc"], errors="coerce").fillna(0).max())
    if "n_icdsc_measurements" in sub.columns:
        out["n_icdsc_measurements_sum"] = float(
            pd.to_numeric(sub["n_icdsc_measurements"], errors="coerce").fillna(0).sum()
        )
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


def _plot_dataset_sizes(sizes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(sizes))
    w = 0.35
    ax.bar(x - w / 2, sizes["n_rows"], width=w, label="Rows", color="#2563eb")
    ax.bar(x + w / 2, sizes["n_unique_patient_ids"], width=w, label="Unique patient IDs", color="#38bdf8")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes["dataset"], rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Dataset sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overlap_bars(rows: List[Tuple[str, int]], title: str, out_path: Path) -> None:
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(labels, vals, color="#059669")
    ax.set_ylabel("Unique patient IDs")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_unmatched(labels: List[str], vals: List[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, vals, color="#dc2626")
    ax.set_ylabel("Patient count")
    ax.set_title("Matching gaps (unique IDs)")
    ax.tick_params(axis="x", rotation=15)
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


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))

    berichte_path = BERICHTE_INPUT_PATH
    baseline_path = STRUCTURED_BASELINE_PATH

    berichte = _load_berichte(berichte_path)
    baseline = _load_baseline(baseline_path)

    DATA_COVERAGE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_COVERAGE_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_COVERAGE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    b_ids = _id_set_berichte(berichte)
    m_ids = _id_set_baseline(baseline)

    inter = b_ids & m_ids
    ber_only = b_ids - m_ids
    base_only = m_ids - b_ids
    union_ids = b_ids | m_ids

    sizes_df = pd.DataFrame(
        [
            {
                "dataset": "Berichte.csv",
                "n_rows": len(berichte),
                "n_unique_patient_ids": len(b_ids),
            },
            {
                "dataset": "structured_baseline.csv",
                "n_rows": len(baseline),
                "n_unique_patient_ids": len(m_ids),
            },
        ]
    )
    sizes_df.to_csv(DATA_COVERAGE_TABLES_DIR / "dataset_sizes.csv", index=False)

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
        if "n_icdsc_measurements" in sub.columns:
            nm = pd.to_numeric(sub["n_icdsc_measurements"], errors="coerce").fillna(0)
            overlap_rows.append(
                {
                    "category": "in_overlap_with_icdsc_measurements_gt_0",
                    "n_unique_patient_ids": int((nm > 0).sum()),
                }
            )

    pd.DataFrame(overlap_rows).to_csv(DATA_COVERAGE_TABLES_DIR / "overlap_counts.csv", index=False)

    dup_base_ids = baseline["PatientenID"].value_counts()
    n_dup_base = int((dup_base_ids > 1).sum())
    if n_dup_base:
        LOGGER.warning(
            "structured_baseline.csv: %d PatientenID values appear on more than one row; "
            "using max(has_delir_icd10), max(max_icdsc), sum(n_icdsc_measurements) per ID.",
            n_dup_base,
        )

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
        nm = agg["n_icdsc_measurements_sum"]
        if mx is not None and (mx > 0 or nm > 0):
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
            "definition": "PatientenID in structured_baseline.csv absent from Berichte.csv.",
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
            "definition": "Berichte patient IDs with max_icdsc>0 OR n_icdsc_measurements>0 (after duplicate-ID aggregation).",
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

    ber_vc = berichte.groupby("PatientID", dropna=False).size().value_counts().sort_index()
    base_vc = baseline.groupby("PatientenID", dropna=False).size().value_counts().sort_index()

    _plot_dataset_sizes(sizes_df, DATA_COVERAGE_PLOTS_DIR / "dataset_sizes.png")

    overlap_plot_data = [
        ("Intersection", len(inter)),
        ("Berichte only", len(ber_only)),
        ("Baseline only", len(base_only)),
    ]
    _plot_overlap_bars(overlap_plot_data, "Patient ID overlap", DATA_COVERAGE_PLOTS_DIR / "overlap_distribution.png")

    linkage_rows = [r for r in unmatched_rows if r["metric_group"] == "linkage"]
    um_labels = [
        "Berichte w/o\nbaseline ID",
        "Baseline w/o\nBerichte ID",
        "Berichte with\nbaseline ID",
        "All PatientID\noverlap",
    ]
    um_vals = [int(r["n_unique_patient_ids"]) for r in linkage_rows]
    _plot_unmatched(um_labels, um_vals, DATA_COVERAGE_PLOTS_DIR / "unmatched_counts.png")

    _plot_duplicates_histogram(ber_vc, base_vc, DATA_COVERAGE_PLOTS_DIR / "duplicates_histogram.png")

    report_lines = [
        "Data coverage analysis (pre-model)",
        "",
        f"Berichte path: {berichte_path}",
        f"Baseline path: {baseline_path}",
        "",
        "Dataset sizes",
        f"  Berichte rows: {len(berichte)}, unique PatientID: {len(b_ids)}",
        f"  Baseline rows: {len(baseline)}, unique PatientenID: {len(m_ids)}",
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
        "Clinical positivity among Berichte patient IDs",
    ]
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

    print(f"Wrote tables under: {DATA_COVERAGE_TABLES_DIR}")
    print(f"Wrote plots under: {DATA_COVERAGE_PLOTS_DIR}")
    print(f"Wrote report: {DATA_COVERAGE_ANALYSIS_DIR / 'report.txt'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
