"""
ICD vs ICDSC cohort overlap analysis.

Independent analysis module:
- Reads raw ICD and ICDSC cohorts
- Computes patient-level overlap metrics
- Writes professional tables, plots, and a report
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import ICD10_PATH, ICDSC_PATH, OUTPUTS_DIR


ICD_PATH = ICD10_PATH
ICDSC_PATH = ICDSC_PATH

ANALYSIS_DIR = OUTPUTS_DIR / "analysis" / "icd_icdsc_overlap"
TABLES_DIR = ANALYSIS_DIR / "tables"
PLOTS_DIR = ANALYSIS_DIR / "plots"


def _mpl_config_dir() -> Path:
    cfg = OUTPUTS_DIR / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def normalize_patient_id(series: pd.Series) -> pd.Series:
    """Normalize patient IDs while preserving their string representation."""
    s = series.astype(str).str.strip()
    s = s[(s.str.len() > 0) & (s.str.lower() != "nan")]
    return s


def _load_input(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _extract_patient_ids(df: pd.DataFrame, label: str) -> pd.Series:
    if "PatientID" not in df.columns:
        raise ValueError(f"{label} must contain 'PatientID'. Found: {list(df.columns)}")
    return normalize_patient_id(df["PatientID"])


def _build_metrics(
    icd_df: pd.DataFrame,
    icdsc_df: pd.DataFrame,
    icd_ids: Set[str],
    icdsc_ids: Set[str],
    overlap: Set[str],
    icd_only: Set[str],
    icdsc_only: Set[str],
) -> Dict[str, float]:
    def _pct(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    return {
        "icd_rows": float(len(icd_df)),
        "icd_unique_patients": float(len(icd_ids)),
        "icdsc_rows": float(len(icdsc_df)),
        "icdsc_unique_patients": float(len(icdsc_ids)),
        "overlap_patients": float(len(overlap)),
        "icd_only_patients": float(len(icd_only)),
        "icdsc_only_patients": float(len(icdsc_only)),
        "overlap_pct_of_icd": _pct(len(overlap), len(icd_ids)),
        "overlap_pct_of_icdsc": _pct(len(overlap), len(icdsc_ids)),
    }


def build_combined_cohort_flags(
    patient_ids: Set[str],
    icd_ids: Set[str],
    icdsc_ids: Set[str],
) -> pd.DataFrame:
    """
    Future-ready helper for combined cohort analyses.

    Not used in current evaluation pipeline; only exported for reuse.
    """
    rows: List[Dict[str, object]] = []
    for pid in sorted(patient_ids):
        has_icd = pid in icd_ids
        has_icdsc = pid in icdsc_ids
        rows.append(
            {
                "PatientID": pid,
                "has_icd": int(has_icd),
                "has_icdsc": int(has_icdsc),
                "has_any_signal": int(has_icd or has_icdsc),
                "has_both_signals": int(has_icd and has_icdsc),
                "is_icd_only": int(has_icd and not has_icdsc),
                "is_icdsc_only": int(has_icdsc and not has_icd),
            }
        )
    return pd.DataFrame(rows)


def _plot_bar(icd_n: int, icdsc_n: int, overlap_n: int, out_path: Path) -> None:
    labels = ["ICD unique patients", "ICDSC unique patients", "Overlap patients"]
    values = [icd_n, icdsc_n, overlap_n]
    colors = ["#1d4ed8", "#7c3aed", "#16a34a"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Unique patient IDs")
    ax.set_title("ICD vs ICDSC patient cohort overlap\nOverlap shown as shared patient IDs across both cohorts")
    ax.tick_params(axis="x", rotation=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_donut(overlap_n: int, only_n: int, title: str, labels: List[str], out_path: Path) -> None:
    values = [overlap_n, only_n]
    colors = ["#16a34a", "#ea580c"]
    total = max(sum(values), 1)

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    wedges, _texts, _autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=_autopct,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_venn_style(icd_only_n: int, overlap_n: int, icdsc_only_n: int, icd_n: int, icdsc_n: int, out_path: Path) -> None:
    values = [icd_only_n, overlap_n, icdsc_only_n]
    labels = ["ICD only", "Overlap", "ICDSC only"]
    colors = ["#ea580c", "#16a34a", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    left = 0
    for value, label, color in zip(values, labels, colors):
        ax.barh(["Patient cohort space"], [value], left=left, color=color, height=0.55, label=label)
        ax.text(left + value / 2.0, 0, f"{label}\n{value}", ha="center", va="center", fontsize=9)
        left += value
    pct_icd = (overlap_n / icd_n * 100.0) if icd_n else 0.0
    pct_icdsc = (overlap_n / icdsc_n * 100.0) if icdsc_n else 0.0
    ax.set_title("ICD and ICDSC cohort overlap schematic")
    ax.set_xlabel("Unique patient IDs")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22))
    ax.text(
        0.01,
        -0.28,
        f"Overlap as % of ICD cohort: {pct_icd:.1f}% | Overlap as % of ICDSC cohort: {pct_icdsc:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heatmap(icd_only_n: int, overlap_n: int, icdsc_only_n: int, out_path: Path) -> None:
    # Rows: ICD present no/yes, Cols: ICDSC present no/yes
    matrix = np.array(
        [
            [0, icdsc_only_n],
            [icd_only_n, overlap_n],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ICDSC: no", "ICDSC: yes"])
    ax.set_yticklabels(["ICD: no", "ICD: yes"])
    ax.set_title("ICD vs ICDSC overlap presence matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="black", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Patient count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _report_lines(metrics: Dict[str, float]) -> List[str]:
    icd_rows = int(metrics["icd_rows"])
    icd_unique = int(metrics["icd_unique_patients"])
    icdsc_rows = int(metrics["icdsc_rows"])
    icdsc_unique = int(metrics["icdsc_unique_patients"])
    overlap = int(metrics["overlap_patients"])
    icd_only = int(metrics["icd_only_patients"])
    icdsc_only = int(metrics["icdsc_only_patients"])
    overlap_pct_icd = metrics["overlap_pct_of_icd"] * 100.0
    overlap_pct_icdsc = metrics["overlap_pct_of_icdsc"] * 100.0

    return [
        "ICD vs ICDSC cohort overlap analysis",
        "",
        f"ICD path: {ICD_PATH}",
        f"ICDSC path: {ICDSC_PATH}",
        "",
        "Counts",
        f"  ICD total rows: {icd_rows}",
        f"  ICD unique PatientIDs: {icd_unique}",
        f"  ICDSC total rows: {icdsc_rows}",
        f"  ICDSC unique PatientIDs: {icdsc_unique}",
        f"  ICDSC column ICDSC_Max present: {metrics.get('icdsc_has_icdsc_max', 0):.0f}",
        "",
        "Overlap",
        f"  overlap patients: {overlap}",
        f"  ICD-only patients: {icd_only}",
        f"  ICDSC-only patients: {icdsc_only}",
        "",
        "Percentages",
        f"  overlap percentage relative to ICD: {overlap_pct_icd:.1f}%",
        f"  overlap percentage relative to ICDSC: {overlap_pct_icdsc:.1f}%",
        "",
        "Interpretation",
        "  ICD and ICDSC do not cover identical patient cohorts. Therefore baseline definitions based on ICD-only, ICDSC-only, or combined criteria may evaluate different patient populations.",
        "  Overlap relative to ICD quantifies how much of the ICD-defined cohort is captured by ICDSC.",
        "  Overlap relative to ICDSC quantifies how much of the ICDSC-defined cohort is captured by ICD.",
        "",
        f"Tables: {TABLES_DIR}",
        f"Plots: {PLOTS_DIR}",
    ]


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    icd_df = _load_input(ICD_PATH, "ICD.csv")
    icdsc_df = _load_input(ICDSC_PATH, "ICDSC.csv")

    icd_ids = set(_extract_patient_ids(icd_df, "ICD.csv").unique())
    icdsc_ids = set(_extract_patient_ids(icdsc_df, "ICDSC.csv").unique())

    overlap_ids = icd_ids & icdsc_ids
    icd_only_ids = icd_ids - icdsc_ids
    icdsc_only_ids = icdsc_ids - icd_ids

    metrics = _build_metrics(
        icd_df=icd_df,
        icdsc_df=icdsc_df,
        icd_ids=icd_ids,
        icdsc_ids=icdsc_ids,
        overlap=overlap_ids,
        icd_only=icd_only_ids,
        icdsc_only=icdsc_only_ids,
    )
    metrics["icdsc_has_icdsc_max"] = float("ICDSC_Max" in icdsc_df.columns)

    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()],
        columns=["metric", "value"],
    )
    summary_df.to_csv(TABLES_DIR / "cohort_overlap_summary.csv", index=False)
    pd.DataFrame({"PatientID": sorted(icd_only_ids)}).to_csv(TABLES_DIR / "icd_only_patient_ids.csv", index=False)
    pd.DataFrame({"PatientID": sorted(icdsc_only_ids)}).to_csv(TABLES_DIR / "icdsc_only_patient_ids.csv", index=False)
    pd.DataFrame({"PatientID": sorted(overlap_ids)}).to_csv(TABLES_DIR / "overlap_patient_ids.csv", index=False)

    _plot_bar(
        icd_n=int(metrics["icd_unique_patients"]),
        icdsc_n=int(metrics["icdsc_unique_patients"]),
        overlap_n=int(metrics["overlap_patients"]),
        out_path=PLOTS_DIR / "cohort_overlap_bar.png",
    )
    _plot_donut(
        overlap_n=int(metrics["overlap_patients"]),
        only_n=int(metrics["icd_only_patients"]),
        title="ICD cohort overlap composition",
        labels=["Overlap with ICDSC", "ICD only"],
        out_path=PLOTS_DIR / "cohort_overlap_pie_icd.png",
    )
    _plot_donut(
        overlap_n=int(metrics["overlap_patients"]),
        only_n=int(metrics["icdsc_only_patients"]),
        title="ICDSC cohort overlap composition",
        labels=["Overlap with ICD", "ICDSC only"],
        out_path=PLOTS_DIR / "cohort_overlap_pie_icdsc.png",
    )
    _plot_venn_style(
        icd_only_n=int(metrics["icd_only_patients"]),
        overlap_n=int(metrics["overlap_patients"]),
        icdsc_only_n=int(metrics["icdsc_only_patients"]),
        icd_n=int(metrics["icd_unique_patients"]),
        icdsc_n=int(metrics["icdsc_unique_patients"]),
        out_path=PLOTS_DIR / "cohort_overlap_venn_style.png",
    )
    _plot_heatmap(
        icd_only_n=int(metrics["icd_only_patients"]),
        overlap_n=int(metrics["overlap_patients"]),
        icdsc_only_n=int(metrics["icdsc_only_patients"]),
        out_path=PLOTS_DIR / "cohort_overlap_heatmap.png",
    )

    (ANALYSIS_DIR / "report.txt").write_text("\n".join(_report_lines(metrics)) + "\n", encoding="utf-8")

    print(f"Wrote tables under: {TABLES_DIR}")
    print(f"Wrote plots under: {PLOTS_DIR}")
    print(f"Wrote report: {ANALYSIS_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
