import os
from pathlib import Path
from typing import Dict, List

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    ANALYSIS_PLOTS_DIR,
    ANALYSIS_REPORTS_DIR,
    ANALYSIS_TABLES_DIR,
    DIAGNOSIS_INPUT_PATH,
    ICD10_PATH,
    ICDSC_PATH,
    REPORT_VS_BASELINE_PATH,
)
from src.pipeline.tabular_io import read_tabular
from src.preprocessing.diagnosis_mapper import build_patient_level_reports

CLASS_LABELS = [0, 1, 2]


def _load_main_df() -> pd.DataFrame:
    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    for col in ["klasse", "baseline_reference_class", "anzahl_treffer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_load_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_tabular(path)


def _normalize_pid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PatientenID" not in out.columns and "PatientID" in out.columns:
        out = out.rename(columns={"PatientID": "PatientenID"})
    if "PatientenID" in out.columns:
        out["PatientenID"] = out["PatientenID"].astype(str).str.strip()
    return out


def _write_input_quality_tables(main_df: pd.DataFrame, diagnosis_rows: pd.DataFrame, icd10: pd.DataFrame, icdsc: pd.DataFrame) -> None:
    checks: List[Dict[str, str]] = []

    def add(check: str, value: str, detail: str = ""):
        checks.append({"check": check, "value": value, "detail": detail})

    add("comparison_rows", str(len(main_df)))
    add("comparison_unique_patientenid", str(main_df["PatientenID"].nunique() if "PatientenID" in main_df.columns else 0))

    for name, df in [
        ("diagnosis_raw", diagnosis_rows),
        ("icd10_raw", icd10),
        ("icdsc_raw", icdsc),
    ]:
        add(f"{name}_rows", str(len(df)))
        if "PatientenID" in df.columns:
            add(f"{name}_unique_patientenid", str(df["PatientenID"].nunique()))
            missing = int(df["PatientenID"].isna().sum())
            add(f"{name}_missing_patientenid", str(missing))

    pd.DataFrame(checks).to_csv(ANALYSIS_TABLES_DIR / "input_quality_checks.csv", index=False)



def _write_distribution_tables(df: pd.DataFrame) -> None:
    pred_dist = df["klasse"].value_counts(dropna=False).sort_index().rename_axis("klasse").reset_index(name="count")
    pred_dist.to_csv(ANALYSIS_TABLES_DIR / "distribution_predicted_class.csv", index=False)

    if "baseline_reference_class" in df.columns:
        ref_dist = (
            df["baseline_reference_class"].value_counts(dropna=False).sort_index().rename_axis("baseline_reference_class").reset_index(name="count")
        )
        ref_dist.to_csv(ANALYSIS_TABLES_DIR / "distribution_reference_class.csv", index=False)

    if "signalstaerke" in df.columns:
        ss = df["signalstaerke"].fillna("missing").value_counts().rename_axis("signalstaerke").reset_index(name="count")
        ss.to_csv(ANALYSIS_TABLES_DIR / "distribution_signalstaerke.csv", index=False)



def _write_patient_level_analysis(df: pd.DataFrame) -> None:
    out = pd.DataFrame()
    out["PatientenID"] = df.get("PatientenID", pd.Series(dtype=str))
    if "anzahl_treffer" in df.columns:
        out["anzahl_treffer"] = pd.to_numeric(df["anzahl_treffer"], errors="coerce").fillna(0)
    else:
        out["anzahl_treffer"] = 0
    out["klasse"] = pd.to_numeric(df.get("klasse", pd.Series(dtype=float)), errors="coerce")
    out["baseline_reference_class"] = pd.to_numeric(df.get("baseline_reference_class", pd.Series(dtype=float)), errors="coerce")
    out["abs_class_error"] = (out["klasse"] - out["baseline_reference_class"]).abs()

    if "delir_signale" in df.columns:
        out["n_signals_from_text"] = (
            df["delir_signale"].fillna("").astype(str).apply(lambda s: 0 if s.strip() == "" else len([x for x in s.split(" | ") if x.strip()]))
        )

    if "report_text" in df.columns:
        out["report_char_len"] = df["report_text"].fillna("").astype(str).str.len()

    out.to_csv(ANALYSIS_TABLES_DIR / "patient_level_analysis_table.csv", index=False)



def _write_confusions(df: pd.DataFrame) -> None:
    if "baseline_reference_class" not in df.columns:
        return

    ct = pd.crosstab(df["baseline_reference_class"], df["klasse"], dropna=False)
    ct = ct.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)
    ct.to_csv(ANALYSIS_TABLES_DIR / "confusion_matrix_3class_analysis.csv")

    row_sums = ct.sum(axis=1).replace(0, np.nan)
    ct_norm = ct.div(row_sums, axis=0).fillna(0)
    ct_norm.to_csv(ANALYSIS_TABLES_DIR / "confusion_matrix_3class_row_normalized.csv")

    errors = df.copy()
    errors["class_error"] = pd.to_numeric(errors["klasse"], errors="coerce") - pd.to_numeric(
        errors["baseline_reference_class"], errors="coerce"
    )
    errors["error_type"] = errors["class_error"].map(lambda v: "overcall" if v > 0 else ("undercall" if v < 0 else "exact"))
    errors.to_csv(ANALYSIS_TABLES_DIR / "class_error_detail_table.csv", index=False)



def _plot_class_distribution(df: pd.DataFrame) -> None:
    pred = df["klasse"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    ref = df.get("baseline_reference_class", pd.Series(dtype=float)).value_counts().reindex(CLASS_LABELS, fill_value=0)

    x = np.arange(len(CLASS_LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, pred.values, width=width, label="prediction", color="#2E86AB")
    ax.bar(x + width / 2, ref.values, width=width, label="reference", color="#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLASS_LABELS])
    ax.set_xlabel("Class")
    ax.set_ylabel("Patients")
    ax.set_title("Predicted vs Reference Class Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_PLOTS_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)



def _plot_signal_strength_by_class(df: pd.DataFrame) -> None:
    if "signalstaerke" not in df.columns:
        return
    tab = pd.crosstab(df["klasse"], df["signalstaerke"], dropna=False)
    tab = tab.sort_index()
    tab.to_csv(ANALYSIS_TABLES_DIR / "signalstaerke_by_predicted_class.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Count")
    ax.set_title("Signal Strength Composition by Predicted Class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_PLOTS_DIR / "02_signalstaerke_by_class.png", dpi=150)
    plt.close(fig)



def _plot_confusion_heatmap(df: pd.DataFrame) -> None:
    if "baseline_reference_class" not in df.columns:
        return
    cm = pd.crosstab(df["baseline_reference_class"], df["klasse"], dropna=False)
    cm = cm.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reference")
    ax.set_title("Confusion Matrix (3-class)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(ANALYSIS_PLOTS_DIR / "03_confusion_heatmap.png", dpi=150)
    plt.close(fig)



def _plot_hits_and_error(df: pd.DataFrame) -> None:
    if "anzahl_treffer" not in df.columns or "baseline_reference_class" not in df.columns:
        return
    x = pd.to_numeric(df["anzahl_treffer"], errors="coerce").fillna(0)
    y = pd.to_numeric(df["klasse"], errors="coerce") - pd.to_numeric(df["baseline_reference_class"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(x, y, alpha=0.7, s=40, color="#1b9e77")
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Number of extracted hits (anzahl_treffer)")
    ax.set_ylabel("Class error (prediction - reference)")
    ax.set_title("Error vs Extracted Hits")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_PLOTS_DIR / "04_error_vs_hits_scatter.png", dpi=150)
    plt.close(fig)



def _plot_report_length(df: pd.DataFrame, reports: pd.DataFrame) -> None:
    if reports.empty:
        return
    rep = reports.copy()
    rep["report_char_len"] = rep["report_text"].fillna("").astype(str).str.len()

    merged = df.merge(rep[["PatientenID", "report_char_len"]], on="PatientenID", how="left") if "PatientenID" in df.columns else rep

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = merged["report_char_len"].dropna()
    if len(vals) > 0:
        ax.hist(vals, bins=min(20, max(5, int(len(vals) / 2))), color="#386cb0", alpha=0.85)
    ax.set_xlabel("Report length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Patient-level Report Length")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_PLOTS_DIR / "05_report_length_histogram.png", dpi=150)
    plt.close(fig)

    merged[["PatientenID", "report_char_len"]].to_csv(ANALYSIS_TABLES_DIR / "report_length_per_patient.csv", index=False)



def _write_text_summary(main_df: pd.DataFrame, diagnosis_raw: pd.DataFrame, icd10_raw: pd.DataFrame, icdsc_raw: pd.DataFrame) -> None:
    lines = []
    lines.append("=== In-depth Analysis Summary ===")
    lines.append("")
    lines.append(f"Rows in comparison table: {len(main_df)}")
    if "PatientenID" in main_df.columns:
        lines.append(f"Unique patients in comparison table: {main_df['PatientenID'].nunique()}")

    lines.append("")
    lines.append("-- Raw source rows --")
    lines.append(f"Diagnosis raw rows: {len(diagnosis_raw)}")
    lines.append(f"ICD10 raw rows: {len(icd10_raw)}")
    lines.append(f"ICDSC raw rows: {len(icdsc_raw)}")

    if "klasse" in main_df.columns:
        lines.append("")
        lines.append("-- Predicted class distribution --")
        dist = main_df["klasse"].value_counts().sort_index().to_dict()
        lines.append(str(dist))

    if "baseline_reference_class" in main_df.columns:
        lines.append("")
        lines.append("-- Reference class distribution --")
        ref_dist = main_df["baseline_reference_class"].value_counts().sort_index().to_dict()
        lines.append(str(ref_dist))

        agree = (
            (pd.to_numeric(main_df["klasse"], errors="coerce") == pd.to_numeric(main_df["baseline_reference_class"], errors="coerce"))
            .mean()
        )
        lines.append(f"Exact class agreement: {agree:.4f}")

    lines.append("")
    lines.append("Generated artifacts:")
    lines.append(f"- tables: {ANALYSIS_TABLES_DIR}")
    lines.append(f"- plots: {ANALYSIS_PLOTS_DIR}")

    (ANALYSIS_REPORTS_DIR / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    ANALYSIS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    main_df = _load_main_df()
    diagnosis_raw = _normalize_pid(_safe_load_tabular(DIAGNOSIS_INPUT_PATH))
    icd10_raw = _normalize_pid(_safe_load_tabular(ICD10_PATH))
    icdsc_raw = _normalize_pid(_safe_load_tabular(ICDSC_PATH))
    reports = build_patient_level_reports()

    # Save standardized source snapshots for traceability
    diagnosis_raw.to_csv(ANALYSIS_TABLES_DIR / "diagnosis_raw_snapshot.csv", index=False)
    icd10_raw.to_csv(ANALYSIS_TABLES_DIR / "icd10_raw_snapshot.csv", index=False)
    icdsc_raw.to_csv(ANALYSIS_TABLES_DIR / "icdsc_raw_snapshot.csv", index=False)
    reports.to_csv(ANALYSIS_TABLES_DIR / "patient_level_reports_snapshot.csv", index=False)

    _write_input_quality_tables(main_df, diagnosis_raw, icd10_raw, icdsc_raw)
    _write_distribution_tables(main_df)
    _write_patient_level_analysis(main_df)
    _write_confusions(main_df)

    _plot_class_distribution(main_df)
    _plot_signal_strength_by_class(main_df)
    _plot_confusion_heatmap(main_df)
    _plot_hits_and_error(main_df)
    _plot_report_length(main_df, reports)

    _write_text_summary(main_df, diagnosis_raw, icd10_raw, icdsc_raw)

    print(f"Analysis tables: {ANALYSIS_TABLES_DIR}")
    print(f"Analysis plots:  {ANALYSIS_PLOTS_DIR}")
    print(f"Analysis report: {ANALYSIS_REPORTS_DIR / 'analysis_summary.txt'}")


if __name__ == "__main__":
    main()
