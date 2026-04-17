import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    ANALYSIS_DIR,
    DIAGNOSIS_INPUT_PATH,
    PATIENT_LEVEL_REPORTS_PATH,
    EXPLORATION_PLOTS_DIR,
    EXPLORATION_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
    PREDICTIONS_DIR,
    ICD10_PATH,
    ICDSC_PATH,
)
from src.pipeline.tabular_io import read_tabular

STOPWORDS = {
    "und", "oder", "bei", "der", "die", "das", "mit", "auf", "von", "ein", "eine",
    "ist", "im", "in", "zu", "zur", "zum", "den", "des", "dem", "als", "nach",
    "ohne", "durch", "nicht", "keine", "klinisch", "patient", "patientin", "status",
}

SIGNAL_CATEGORY_PATTERNS = {
    "disorientation": [r"desorient", r"orientier"],
    "explicit_delirium": [r"\bdelir\b", r"delirium"],
    "agitation_hyperactivity": [r"agitation", r"unruh", r"hyperaktiv"],
    "vigilance": [r"vigil", r"somnol", r"bewusst", r"schlaef"],
    "delirium_therapy": [r"haloperidol", r"quetiapin", r"risperidon", r"olanzapin"],
    "delirium_prophylaxis": [r"prophyl", r"melatonin", r"reorient"],
}

PREDICTIONS_PROMPT_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"
EXPLORATION_REPORT_PATH = ANALYSIS_DIR / "exploration" / "report.txt"


def _normalize_pid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PatientenID" not in out.columns and "PatientID" in out.columns:
        out = out.rename(columns={"PatientID": "PatientenID"})
    if "PatientenID" in out.columns:
        out["PatientenID"] = out["PatientenID"].astype(str).str.strip()
    return out


def _safe_load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _normalize_pid(read_tabular(path))


def _missingness_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"dataset": name, "column": "<none>", "missing_count": 0, "missing_rate": 0.0}])
    rows = []
    n = len(df)
    for c in df.columns:
        m = int(df[c].isna().sum())
        rows.append({"dataset": name, "column": c, "missing_count": m, "missing_rate": round(m / n, 6) if n else 0.0})
    return pd.DataFrame(rows)


def _tokenize(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for t in texts:
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{2,}", str(t).lower())
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) >= 4]
        counter.update(tokens)
    return counter


def _load_predictions() -> pd.DataFrame:
    for candidate in [REPORT_VS_BASELINE_PATH, PREDICTIONS_PROMPT_PATH]:
        if candidate.exists():
            return _normalize_pid(pd.read_csv(candidate))
    return pd.DataFrame()


def _write_overview_tables(diag: pd.DataFrame, icd10: pd.DataFrame, icdsc: pd.DataFrame) -> None:
    overview = pd.DataFrame(
        [
            {"dataset": "diagnosis", "rows": len(diag), "columns": len(diag.columns), "unique_patients": diag["PatientenID"].nunique() if "PatientenID" in diag.columns else 0},
            {"dataset": "icd10", "rows": len(icd10), "columns": len(icd10.columns), "unique_patients": icd10["PatientenID"].nunique() if "PatientenID" in icd10.columns else 0},
            {"dataset": "icdsc", "rows": len(icdsc), "columns": len(icdsc.columns), "unique_patients": icdsc["PatientenID"].nunique() if "PatientenID" in icdsc.columns else 0},
        ]
    )
    overview.to_csv(EXPLORATION_TABLES_DIR / "dataset_overview.csv", index=False)

    miss = pd.concat(
        [
            _missingness_table("diagnosis", diag),
            _missingness_table("icd10", icd10),
            _missingness_table("icdsc", icdsc),
        ],
        ignore_index=True,
    )
    miss.to_csv(EXPLORATION_TABLES_DIR / "missingness_by_dataset.csv", index=False)

    if "PatientenID" in diag.columns and "PatientenID" in icd10.columns and "PatientenID" in icdsc.columns:
        diag_ids = set(diag["PatientenID"])
        icd10_ids = set(icd10["PatientenID"])
        icdsc_ids = set(icdsc["PatientenID"])
        set_rows = [
            {"set_name": "diagnosis_only", "count": len(diag_ids - icd10_ids - icdsc_ids)},
            {"set_name": "icd10_only", "count": len(icd10_ids - diag_ids - icdsc_ids)},
            {"set_name": "icdsc_only", "count": len(icdsc_ids - diag_ids - icd10_ids)},
            {"set_name": "intersection_all_three", "count": len(diag_ids & icd10_ids & icdsc_ids)},
            {"set_name": "diag_not_in_icd10", "count": len(diag_ids - icd10_ids)},
            {"set_name": "diag_not_in_icdsc", "count": len(diag_ids - icdsc_ids)},
        ]
        pd.DataFrame(set_rows).to_csv(EXPLORATION_TABLES_DIR / "patient_set_overlap_summary.csv", index=False)


def _plot_report_length_distribution(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty or "report_text" not in reports.columns:
        return pd.DataFrame(columns=["PatientenID", "report_characters", "report_words"])

    out = reports.copy()
    out["report_text"] = out["report_text"].fillna("").astype(str)
    out["report_characters"] = out["report_text"].str.len()
    out["report_words"] = out["report_text"].str.split().str.len()
    out[["PatientenID", "report_characters", "report_words"]].to_csv(
        EXPLORATION_TABLES_DIR / "report_length_distribution.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = out["report_words"].dropna()
    if len(vals) > 0:
        bins = min(30, max(8, int(np.sqrt(len(vals)))))
        ax.hist(vals, bins=bins, color="#2E86AB", alpha=0.85)
    ax.set_title("Patient Report Length Distribution")
    ax.set_xlabel("Words per patient-level report")
    ax.set_ylabel("Number of reports")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "01_report_length_distribution.png", dpi=300)
    plt.close(fig)
    return out[["PatientenID", "report_characters", "report_words"]]


def _plot_keyword_frequency_from_reports(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty or "report_text" not in reports.columns:
        return pd.DataFrame(columns=["term", "count"])

    term_counter = _tokenize(reports["report_text"].fillna("").astype(str).tolist())
    top_terms = pd.DataFrame(term_counter.most_common(100), columns=["term", "count"])
    top_terms.to_csv(EXPLORATION_TABLES_DIR / "keyword_frequency_top100.csv", index=False)

    plot_terms = top_terms.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    if not plot_terms.empty:
        ax.barh(plot_terms["term"], plot_terms["count"], color="#355C7D")
    ax.set_title("Top 20 Keywords in Patient Reports")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Keyword")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "02_keyword_frequency_top20.png", dpi=300)
    plt.close(fig)
    return top_terms


def _plot_signal_category_frequencies(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "delir_signale" not in predictions.columns:
        return pd.DataFrame(columns=["signal_category", "count"])

    counts = {name: 0 for name in SIGNAL_CATEGORY_PATTERNS}
    counts["other_signal_mentions"] = 0
    signal_text = predictions["delir_signale"].fillna("").astype(str)

    for row in signal_text:
        parts = [p.strip().lower() for p in row.split("|") if p.strip()]
        for part in parts:
            matched = False
            for category, patterns in SIGNAL_CATEGORY_PATTERNS.items():
                if any(re.search(pattern, part) for pattern in patterns):
                    counts[category] += 1
                    matched = True
            if not matched:
                counts["other_signal_mentions"] += 1

    out = (
        pd.DataFrame(
            [{"signal_category": key, "count": value} for key, value in counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(EXPLORATION_TABLES_DIR / "signal_category_frequency.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(out["signal_category"], out["count"], color="#6C5B7B")
    ax.set_title("Signal Category Frequencies from Agent Extraction")
    ax.set_xlabel("Signal category")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "03_signal_category_frequency.png", dpi=300)
    plt.close(fig)
    return out


def _plot_class_distribution_if_available(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "klasse" not in predictions.columns:
        return pd.DataFrame(columns=["class", "count", "source"])

    pred = pd.to_numeric(predictions["klasse"], errors="coerce").value_counts().sort_index()
    rows = [{"class": int(k), "count": int(v), "source": "prediction"} for k, v in pred.items()]

    if "baseline_reference_class" in predictions.columns:
        ref = pd.to_numeric(predictions["baseline_reference_class"], errors="coerce").value_counts().sort_index()
        rows.extend({"class": int(k), "count": int(v), "source": "baseline"} for k, v in ref.items())

    dist_df = pd.DataFrame(rows).sort_values(["source", "class"])
    dist_df.to_csv(EXPLORATION_TABLES_DIR / "class_distribution_if_available.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not dist_df.empty:
        pivot = (
            dist_df.pivot_table(index="class", columns="source", values="count", aggfunc="sum")
            .fillna(0)
            .sort_index()
        )
        x = np.arange(len(pivot.index))
        width = 0.35
        ax.bar(x - width / 2, pivot.get("prediction", pd.Series(0, index=pivot.index)).values, width=width, label="Prediction")
        if "baseline" in pivot.columns:
            ax.bar(x + width / 2, pivot["baseline"].values, width=width, label="Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in pivot.index])
    ax.set_title("Class Distribution (Available Labels)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "04_class_distribution_if_available.png", dpi=300)
    plt.close(fig)
    return dist_df


def _write_optional_token_frequency(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty or "report_text" not in reports.columns:
        return pd.DataFrame(columns=["token", "count"])
    counter = Counter()
    for text in reports["report_text"].fillna("").astype(str):
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text.lower())
        counter.update(tokens)
    token_df = pd.DataFrame(counter.most_common(200), columns=["token", "count"])
    token_df.to_csv(EXPLORATION_TABLES_DIR / "token_frequency_top200.csv", index=False)
    return token_df


def _diagnosis_exploration(diag: pd.DataFrame) -> None:
    if diag.empty:
        return
    if "Value" in diag.columns:
        top_values = diag["Value"].fillna("").astype(str).value_counts().head(30).rename_axis("value").reset_index(name="count")
        top_values.to_csv(EXPLORATION_TABLES_DIR / "top_diagnosis_entries.csv", index=False)

        term_counter = _tokenize(diag["Value"].fillna("").astype(str).tolist())
        top_terms = pd.DataFrame(term_counter.most_common(50), columns=["term", "count"])
        top_terms.to_csv(EXPLORATION_TABLES_DIR / "top_diagnosis_terms.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_terms = top_terms.head(20).iloc[::-1]
        if not plot_terms.empty:
            ax.barh(plot_terms["term"], plot_terms["count"], color="#355C7D")
        ax.set_title("Top diagnosis terms (Value text)")
        ax.set_xlabel("Frequency")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "10_top_diagnosis_terms.png", dpi=300)
        plt.close(fig)

    if "ParameterID" in diag.columns:
        param_counts = diag["ParameterID"].astype(str).value_counts().rename_axis("ParameterID").reset_index(name="count")
        param_counts.to_csv(EXPLORATION_TABLES_DIR / "parameterid_frequency.csv", index=False)

    if "Time" in diag.columns:
        ts = pd.to_datetime(diag["Time"], errors="coerce")
        tdf = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.day_name()})
        tdf["hour"].value_counts().sort_index().rename_axis("hour").reset_index(name="count").to_csv(
            EXPLORATION_TABLES_DIR / "diagnosis_by_hour.csv", index=False
        )
        wd = tdf["weekday"].value_counts().rename_axis("weekday").reset_index(name="count")
        wd.to_csv(EXPLORATION_TABLES_DIR / "diagnosis_by_weekday.csv", index=False)


def _icd10_exploration(icd10: pd.DataFrame) -> None:
    if icd10.empty:
        return
    if "Code" in icd10.columns:
        codes = icd10["Code"].fillna("").astype(str).str.strip().str.upper()
        code_counts = codes.value_counts().rename_axis("Code").reset_index(name="count")
        code_counts.to_csv(EXPLORATION_TABLES_DIR / "icd10_code_frequency.csv", index=False)

        prefixes = codes.apply(lambda x: x.split(".")[0] if x else "")
        pref_counts = prefixes.value_counts().rename_axis("CodePrefix").reset_index(name="count")
        pref_counts.to_csv(EXPLORATION_TABLES_DIR / "icd10_prefix_frequency.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_codes = code_counts.head(20).iloc[::-1]
        if not plot_codes.empty:
            ax.barh(plot_codes["Code"], plot_codes["count"], color="#6C5B7B")
        ax.set_title("Top ICD10 codes")
        ax.set_xlabel("Frequency")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "11_top_icd10_codes.png", dpi=300)
        plt.close(fig)


def _icdsc_exploration(icdsc: pd.DataFrame) -> None:
    if icdsc.empty:
        return
    out_rows = []
    if "ICDSC_Value" in icdsc.columns:
        vals = pd.to_numeric(icdsc["ICDSC_Value"], errors="coerce")
        out_rows.append({"metric": "icdsc_value_count", "value": int(vals.notna().sum())})
        out_rows.append({"metric": "icdsc_value_mean", "value": float(vals.mean()) if vals.notna().any() else 0.0})
        out_rows.append({"metric": "icdsc_value_median", "value": float(vals.median()) if vals.notna().any() else 0.0})
        out_rows.append({"metric": "icdsc_value_max", "value": float(vals.max()) if vals.notna().any() else 0.0})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(vals.dropna(), bins=np.arange(-0.5, 9.5, 1), color="#C06C84", alpha=0.85, rwidth=0.9)
        ax.set_xticks(range(0, 9))
        ax.set_xlabel("ICDSC_Value")
        ax.set_ylabel("Count")
        ax.set_title("ICDSC Value Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "12_icdsc_value_histogram.png", dpi=300)
        plt.close(fig)

    if "ICDSC_DelirFlag" in icdsc.columns:
        flags = pd.to_numeric(icdsc["ICDSC_DelirFlag"], errors="coerce").fillna(0).astype(int)
        flag_counts = flags.value_counts().sort_index().rename_axis("ICDSC_DelirFlag").reset_index(name="count")
        flag_counts.to_csv(EXPLORATION_TABLES_DIR / "icdsc_flag_frequency.csv", index=False)
        out_rows.append({"metric": "icdsc_delir_flag_rate", "value": float((flags == 1).mean())})

    if "ICDSC_Time" in icdsc.columns:
        ts = pd.to_datetime(icdsc["ICDSC_Time"], errors="coerce")
        pd.DataFrame({"hour": ts.dt.hour}).value_counts().sort_index().rename_axis("hour").reset_index(name="count").to_csv(
            EXPLORATION_TABLES_DIR / "icdsc_by_hour.csv", index=False
        )

    if out_rows:
        pd.DataFrame(out_rows).to_csv(EXPLORATION_TABLES_DIR / "icdsc_summary_metrics.csv", index=False)


def _patient_activity_tables(diag: pd.DataFrame, icd10: pd.DataFrame, icdsc: pd.DataFrame) -> None:
    frames = []
    for name, df in [("diagnosis", diag), ("icd10", icd10), ("icdsc", icdsc)]:
        if "PatientenID" in df.columns and not df.empty:
            c = df.groupby("PatientenID").size().reset_index(name=f"{name}_rows")
            frames.append(c)

    if not frames:
        return

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="PatientenID", how="outer")
    merged = merged.fillna(0)
    merged.to_csv(EXPLORATION_TABLES_DIR / "patient_activity_across_sources.csv", index=False)

    numeric_cols = [c for c in merged.columns if c.endswith("_rows")]
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        merged[numeric_cols].boxplot(ax=ax)
        ax.set_title("Per-patient row count distribution by source")
        ax.set_ylabel("Rows per patient")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "13_patient_activity_boxplot.png", dpi=300)
        plt.close(fig)


def _write_exploration_summary(
    diag: pd.DataFrame,
    icd10: pd.DataFrame,
    icdsc: pd.DataFrame,
    reports: pd.DataFrame,
    predictions: pd.DataFrame,
    top_terms: pd.DataFrame,
    signal_freq: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Thesis-Level Input Data Exploration Report")
    lines.append("")
    lines.append(f"Diagnosis rows: {len(diag)}")
    lines.append(f"ICD10 rows: {len(icd10)}")
    lines.append(f"ICDSC rows: {len(icdsc)}")
    lines.append(f"Patient-level reports: {len(reports)}")
    lines.append(f"Prediction rows available: {len(predictions)}")
    lines.append("")

    if "PatientenID" in diag.columns:
        lines.append(f"Unique diagnosis patients: {diag['PatientenID'].nunique()}")
    if "PatientenID" in icd10.columns:
        lines.append(f"Unique ICD10 patients: {icd10['PatientenID'].nunique()}")
    if "PatientenID" in icdsc.columns:
        lines.append(f"Unique ICDSC patients: {icdsc['PatientenID'].nunique()}")

    if not top_terms.empty:
        lines.append("")
        lines.append("Top keywords in patient-level reports:")
        for _, row in top_terms.head(5).iterrows():
            lines.append(f"- {row['term']}: {int(row['count'])}")

    if not signal_freq.empty:
        lines.append("")
        lines.append("Most frequent extracted signal categories:")
        for _, row in signal_freq.head(5).iterrows():
            lines.append(f"- {row['signal_category']}: {int(row['count'])}")

    lines.append("")
    lines.append("Artifacts")
    lines.append(f"- tables: {EXPLORATION_TABLES_DIR}")
    lines.append(f"- plots: {EXPLORATION_PLOTS_DIR}")
    EXPLORATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    EXPLORATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EXPLORATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLORATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    diagnosis = _safe_load(DIAGNOSIS_INPUT_PATH)
    icd10 = _safe_load(ICD10_PATH)
    icdsc = _safe_load(ICDSC_PATH)
    reports = _safe_load(PATIENT_LEVEL_REPORTS_PATH)
    predictions = _load_predictions()

    _write_overview_tables(diagnosis, icd10, icdsc)
    _diagnosis_exploration(diagnosis)
    _icd10_exploration(icd10)
    _icdsc_exploration(icdsc)
    _patient_activity_tables(diagnosis, icd10, icdsc)
    _plot_report_length_distribution(reports)
    top_terms = _plot_keyword_frequency_from_reports(reports)
    signal_freq = _plot_signal_category_frequencies(predictions)
    _plot_class_distribution_if_available(predictions)
    _write_optional_token_frequency(reports)
    _write_exploration_summary(diagnosis, icd10, icdsc, reports, predictions, top_terms, signal_freq)

    print(f"Exploration tables: {EXPLORATION_TABLES_DIR}")
    print(f"Exploration plots:  {EXPLORATION_PLOTS_DIR}")
    print(f"Exploration report: {EXPLORATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
