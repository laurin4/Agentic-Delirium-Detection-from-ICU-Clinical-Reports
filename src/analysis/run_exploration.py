"""EDA over Berichte / ICD / ICDSC and optional prediction merges.

Production inputs: `data/raw/Berichte.csv`, `ICD.csv`, `ICDSC.csv` (no Diagnosenliste.csv).
Report sections ([Diagnosen], [Epikrise], …) come from Berichte columns or stitched report_text.
"""

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

BERICHTE_SECTION_COLUMNS = [
    ("diag", "Diagnosen"),
    ("epikrise", "Epikrise"),
    ("jetziges_leiden", "Jetziges Leiden"),
    ("prozedere", "Prozedere"),
]

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
    BERICHTE_INPUT_PATH,
    DIAGNOSIS_INPUT_PATH,
    EXPLORATION_PLOTS_DIR,
    EXPLORATION_TABLES_DIR,
    ICD10_PATH,
    ICDSC_PATH,
    LEGACY_DIAGNOSIS_INPUT_PATH,
    PATIENT_LEVEL_REPORTS_PATH,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PREDICTIONS_DIR,
    REPORT_VS_BASELINE_PATH,
    STRUCTURED_BASELINE_PATH,
)
from src.analysis.cohort_counts import load_and_compute_current_cohort_counts, print_current_cohort_counts
from src.pipeline.schema_normalize import normalize_icd10_source_columns, normalize_icdsc_source_columns
from src.pipeline.tabular_io import read_tabular
from src.preprocessing.berichte_filters import exclude_dokumentationsblatt, normalize_bertyp
from src.preprocessing.berichte_mapper import build_patient_level_berichte_reports, load_berichte_dataframe
from src.preprocessing.diagnosis_mapper import load_diagnosis_dataframe

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

DELIR_KEYWORDS = ["delir", "desorient", "verwirr", "agitation", "vigilanz"]
AGENT1_SIGNAL_KEYS = [
    "desorientierung",
    "delir_explizit",
    "vigilanz",
    "hyperaktivitaet_agitation",
    "delir_therapie",
    "delir_prophylaxe",
]

PREDICTIONS_PROMPT_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"
EXPLORATION_REPORT_PATH = ANALYSIS_DIR / "exploration" / "report.txt"


def _normalize_pid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PatientenID" not in out.columns and "PatientID" in out.columns:
        out = out.rename(columns={"PatientID": "PatientenID"})
    if "PatientenID" in out.columns:
        out["PatientenID"] = out["PatientenID"].astype(str).str.strip()
    return out


def _safe_load(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return _normalize_pid(read_tabular(path))


def _load_structured_baseline_for_exploration() -> pd.DataFrame:
    """Patient-level baseline from outputs/baseline/structured_baseline.csv (deduplicated)."""
    if not STRUCTURED_BASELINE_PATH.exists():
        return pd.DataFrame()
    from src.analysis.cohort_counts import load_structured_baseline_rows

    return _normalize_pid(load_structured_baseline_rows(STRUCTURED_BASELINE_PATH))


def _warn_legacy_diagnosis_missing() -> None:
    print(
        "Note: Diagnosenliste.csv is deprecated/removed. "
        "Exploration uses Berichte.csv report sections instead."
    )


def _load_reports_for_exploration() -> pd.DataFrame:
    """Patient-level reports from Berichte.csv, with optional prepared CSV fallback."""
    if BERICHTE_INPUT_PATH.exists():
        try:
            return _normalize_pid(build_patient_level_berichte_reports(BERICHTE_INPUT_PATH))
        except Exception as exc:
            LOGGER.warning("Could not build reports from Berichte.csv: %s", exc)
    elif PATIENT_LEVEL_REPORTS_PATH.exists():
        LOGGER.warning("Berichte.csv missing; using prepared patient_level_reports.csv.")
        return _safe_load(PATIENT_LEVEL_REPORTS_PATH)
    else:
        print(f"Warning: no report input at {BERICHTE_INPUT_PATH}; report EDA will be limited.")
    return pd.DataFrame(columns=["PatientenID", "bericht", "report_text"])


def _load_legacy_diagnosis_optional() -> pd.DataFrame:
    """Load legacy Diagnosenliste only when explicitly present (optional)."""
    for candidate in (DIAGNOSIS_INPUT_PATH, LEGACY_DIAGNOSIS_INPUT_PATH):
        if candidate is not None and candidate.exists():
            print(f"Warning: using legacy diagnosis file {candidate} (deprecated).")
            return _normalize_pid(load_diagnosis_dataframe(candidate))
    return pd.DataFrame()


def _load_raw_berichte_optional() -> pd.DataFrame:
    if not BERICHTE_INPUT_PATH.exists():
        return pd.DataFrame()
    try:
        df = _normalize_pid(load_berichte_dataframe(BERICHTE_INPUT_PATH))
        df, excluded = exclude_dokumentationsblatt(df)
        if excluded:
            print(f"excluded_dokumentationsblatt_count={excluded}")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


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


def _get_report_text_series(reports: pd.DataFrame) -> pd.Series:
    if "report_text" in reports.columns:
        return reports["report_text"].fillna("").astype(str)
    if "bericht" in reports.columns:
        return reports["bericht"].fillna("").astype(str)
    return pd.Series(dtype=str)


def _plot_top_diagnoses(reports: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    texts = _get_report_text_series(reports)
    if texts.empty:
        return pd.DataFrame(columns=["diagnosis", "count"])

    counter: Counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)

    top_df = pd.DataFrame(counter.most_common(20), columns=["diagnosis", "count"])
    top_df.to_csv(output_dir / "top_diagnoses.csv", index=False)

    plot_df = top_df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    if not plot_df.empty:
        ax.barh(plot_df["diagnosis"], plot_df["count"], color="#355C7D")
    ax.set_title("Top Diagnoses (All Reports)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Diagnosis")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "top_diagnoses.png", dpi=300)
    plt.close(fig)
    return top_df


def _plot_delir_diagnoses(reports: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    texts = _get_report_text_series(reports)
    if texts.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    counter: Counter = Counter()
    for text in texts:
        lower = text.lower()
        for keyword in DELIR_KEYWORDS:
            count = lower.count(keyword)
            if count > 0:
                counter[keyword] += count

    delir_df = pd.DataFrame(counter.most_common(), columns=["keyword", "count"])
    delir_df.to_csv(output_dir / "top_delir_diagnoses.csv", index=False)

    plot_df = delir_df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if not plot_df.empty:
        ax.barh(plot_df["keyword"], plot_df["count"], color="#6C5B7B")
    ax.set_title("Delir-Related Diagnoses (Keyword Frequency)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Keyword")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "top_delir_diagnoses.png", dpi=300)
    plt.close(fig)
    return delir_df


def _write_overview_tables(
    berichte: pd.DataFrame,
    icd10: pd.DataFrame,
    icdsc: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    def _uniq(df: pd.DataFrame) -> int:
        return int(df["PatientenID"].nunique()) if "PatientenID" in df.columns and not df.empty else 0

    overview = pd.DataFrame(
        [
            {"dataset": "berichte_reports", "rows": len(berichte), "columns": len(berichte.columns), "unique_patients": _uniq(berichte)},
            {"dataset": "icd10", "rows": len(icd10), "columns": len(icd10.columns), "unique_patients": _uniq(icd10)},
            {"dataset": "icdsc", "rows": len(icdsc), "columns": len(icdsc.columns), "unique_patients": _uniq(icdsc)},
            {"dataset": "structured_baseline", "rows": len(baseline), "columns": len(baseline.columns), "unique_patients": _uniq(baseline)},
        ]
    )
    overview.to_csv(EXPLORATION_TABLES_DIR / "dataset_overview.csv", index=False)

    miss = pd.concat(
        [
            _missingness_table("berichte_reports", berichte),
            _missingness_table("icd10", icd10),
            _missingness_table("icdsc", icdsc),
            _missingness_table("structured_baseline", baseline),
        ],
        ignore_index=True,
    )
    miss.to_csv(EXPLORATION_TABLES_DIR / "missingness_by_dataset.csv", index=False)

    if "PatientenID" in berichte.columns and "PatientenID" in icd10.columns and "PatientenID" in icdsc.columns:
        ber_ids = set(berichte["PatientenID"])
        icd10_ids = set(icd10["PatientenID"])
        icdsc_ids = set(icdsc["PatientenID"])
        set_rows = [
            {"set_name": "berichte_only", "count": len(ber_ids - icd10_ids - icdsc_ids)},
            {"set_name": "icd10_only", "count": len(icd10_ids - ber_ids - icdsc_ids)},
            {"set_name": "icdsc_only", "count": len(icdsc_ids - ber_ids - icd10_ids)},
            {"set_name": "intersection_all_three", "count": len(ber_ids & icd10_ids & icdsc_ids)},
            {"set_name": "berichte_not_in_icd10", "count": len(ber_ids - icd10_ids)},
            {"set_name": "berichte_not_in_icdsc", "count": len(ber_ids - icdsc_ids)},
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


def _plot_signal_category_frequencies(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["signal_category", "count"])

    counts = {key: 0 for key in AGENT1_SIGNAL_KEYS}
    if all(key in predictions.columns for key in AGENT1_SIGNAL_KEYS):
        for key in AGENT1_SIGNAL_KEYS:
            series = predictions[key]
            if series.dtype == bool:
                counts[key] = int(series.fillna(False).sum())
            else:
                parsed = series.fillna("").astype(str)
                counts[key] = int(parsed.apply(lambda v: 0 if v.strip() == "" else len([x for x in v.split(" | ") if x.strip()])).sum())
    elif "delir_signale" in predictions.columns:
        # Fallback for current combined extraction text format.
        signal_text = predictions["delir_signale"].fillna("").astype(str)
        for row in signal_text:
            parts = [p.strip().lower() for p in row.split("|") if p.strip()]
            for part in parts:
                for category, patterns in SIGNAL_CATEGORY_PATTERNS.items():
                    if any(re.search(pattern, part) for pattern in patterns):
                        if category == "disorientation":
                            counts["desorientierung"] += 1
                        elif category == "explicit_delirium":
                            counts["delir_explizit"] += 1
                        elif category == "vigilance":
                            counts["vigilanz"] += 1
                        elif category == "agitation_hyperactivity":
                            counts["hyperaktivitaet_agitation"] += 1
                        elif category == "delirium_therapy":
                            counts["delir_therapie"] += 1
                        elif category == "delirium_prophylaxis":
                            counts["delir_prophylaxe"] += 1

    out = pd.DataFrame(
        [{"signal_category": key, "count": int(value)} for key, value in counts.items()]
    ).sort_values("count", ascending=False)
    out.to_csv(output_dir / "signal_category_frequencies.csv", index=False)

    plot_df = out.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if not plot_df.empty:
        ax.barh(plot_df["signal_category"], plot_df["count"], color="#5D6D7E")
    ax.set_title("Delir Signal Category Frequency")
    ax.set_xlabel("Count")
    ax.set_ylabel("Signal category")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "signal_category_frequencies.png", dpi=300)
    plt.close(fig)
    return out


def _plot_class_distribution_if_available(predictions: pd.DataFrame) -> pd.DataFrame:
    """Binary klasse vs optional binary baselines (no multiclass reference)."""
    if predictions.empty or "klasse" not in predictions.columns:
        return pd.DataFrame(columns=["label", "count", "source"])

    pdf = predictions.copy()
    pdf["_k"] = pd.to_numeric(pdf["klasse"], errors="coerce")
    pdf = pdf[pdf["_k"].isin([0, 1])].copy()
    if pdf.empty:
        return pd.DataFrame(columns=["label", "count", "source"])

    rows = []
    vc = pdf["_k"].astype(int).value_counts().sort_index()
    for k, v in vc.items():
        rows.append({"label": f"prediction_klasse_{int(k)}", "count": int(v), "source": "prediction"})

    for base_col in ("baseline_icd10", "baseline_icdsc_ge_4"):
        if base_col not in pdf.columns:
            continue
        b = pd.to_numeric(pdf[base_col], errors="coerce").fillna(0).astype(int)
        pos = int((b == 1).sum())
        neg = int((b == 0).sum())
        rows.append({"label": f"{base_col}_positive", "count": pos, "source": "baseline"})
        rows.append({"label": f"{base_col}_negative", "count": neg, "source": "baseline"})

    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(EXPLORATION_TABLES_DIR / "class_distribution_binary.csv", index=False)

    plot_df = dist_df[dist_df["source"] == "prediction"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not plot_df.empty:
        ax.bar(plot_df["label"], plot_df["count"], color="#2E86AB")
        ax.set_title("Binary prediction class distribution (klasse 0/1)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXPLORATION_PLOTS_DIR / "04_class_distribution_binary.png", dpi=300)
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


def _berichte_sections_exploration(raw_berichte: pd.DataFrame) -> None:
    """Token/length summaries per Berichte section column ([Diagnosen], …)."""
    if raw_berichte.empty:
        return

    section_rows: List[dict] = []
    for col, label in BERICHTE_SECTION_COLUMNS:
        if col not in raw_berichte.columns:
            continue
        texts = raw_berichte[col].fillna("").astype(str)
        non_empty = texts[texts.str.strip().astype(bool)]
        section_rows.append(
            {
                "section": label,
                "column": col,
                "non_empty_rows": int(len(non_empty)),
                "mean_chars": float(non_empty.str.len().mean()) if len(non_empty) else 0.0,
            }
        )
        term_counter = _tokenize(non_empty.tolist())
        top_terms = pd.DataFrame(term_counter.most_common(30), columns=["term", "count"])
        safe_name = col.replace(" ", "_")
        top_terms.to_csv(EXPLORATION_TABLES_DIR / f"berichte_section_{safe_name}_top_terms.csv", index=False)

    if section_rows:
        pd.DataFrame(section_rows).to_csv(EXPLORATION_TABLES_DIR / "berichte_section_overview.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        sdf = pd.DataFrame(section_rows)
        ax.barh(sdf["section"], sdf["non_empty_rows"], color="#355C7D")
        ax.set_title("Berichte.csv: non-empty rows per report section")
        ax.set_xlabel("Row count")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "10_berichte_section_coverage.png", dpi=300)
        plt.close(fig)


def _diagnosis_exploration(diag: pd.DataFrame) -> None:
    """Legacy Diagnosenliste EDA (only when legacy file is present)."""
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
    icd10 = normalize_icd10_source_columns(icd10)
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
    icdsc = normalize_icdsc_source_columns(icdsc)
    out_rows = []
    score_col = "ICDSC_Max" if "ICDSC_Max" in icdsc.columns else None
    if score_col:
        vals = pd.to_numeric(icdsc[score_col], errors="coerce")
        out_rows.append({"metric": "icdsc_max_count", "value": int(vals.notna().sum())})
        out_rows.append({"metric": "icdsc_max_mean", "value": float(vals.mean()) if vals.notna().any() else 0.0})
        out_rows.append({"metric": "icdsc_max_median", "value": float(vals.median()) if vals.notna().any() else 0.0})
        out_rows.append({"metric": "icdsc_max_global_max", "value": float(vals.max()) if vals.notna().any() else 0.0})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(vals.dropna(), bins=np.arange(-0.5, 9.5, 1), color="#C06C84", alpha=0.85, rwidth=0.9)
        ax.set_xticks(range(0, 9))
        ax.set_xlabel("ICDSC_Max")
        ax.set_ylabel("Count")
        ax.set_title("ICDSC_Max Distribution (patient-level)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "12_icdsc_max_histogram.png", dpi=300)
        plt.close(fig)

    if out_rows:
        pd.DataFrame(out_rows).to_csv(EXPLORATION_TABLES_DIR / "icdsc_summary_metrics.csv", index=False)


def _patient_activity_tables(berichte: pd.DataFrame, icd10: pd.DataFrame, icdsc: pd.DataFrame) -> None:
    frames = []
    for name, df in [("berichte", berichte), ("icd10", icd10), ("icdsc", icdsc)]:
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
    icd10: pd.DataFrame,
    icdsc: pd.DataFrame,
    baseline: pd.DataFrame,
    reports: pd.DataFrame,
    predictions: pd.DataFrame,
    top_terms: pd.DataFrame,
    top_diagnoses: pd.DataFrame,
    top_delir_diagnoses: pd.DataFrame,
    signal_freq: pd.DataFrame,
    legacy_diag: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Thesis-Level Input Data Exploration Report")
    lines.append("")
    lines.append("Primary text source: data/raw/Berichte.csv (Diagnosenliste.csv deprecated).")
    lines.append("")
    lines.append(f"Berichte patient-level reports: {len(reports)}")
    lines.append(f"ICD rows: {len(icd10)}")
    lines.append(f"ICDSC rows: {len(icdsc)}")
    lines.append(f"Structured baseline patients: {len(baseline)}")
    lines.append(f"Prediction rows available: {len(predictions)}")
    if not legacy_diag.empty:
        lines.append(f"Legacy diagnosis rows (optional): {len(legacy_diag)}")
    lines.append("")

    if "PatientenID" in reports.columns:
        lines.append(f"Unique Berichte patients: {reports['PatientenID'].nunique()}")
    if "PatientenID" in icd10.columns:
        lines.append(f"Unique ICD patients: {icd10['PatientenID'].nunique()}")
    if "PatientenID" in icdsc.columns:
        lines.append(f"Unique ICDSC patients: {icdsc['PatientenID'].nunique()}")

    if not top_terms.empty:
        lines.append("")
        lines.append("Top keywords in patient-level reports:")
        for _, row in top_terms.head(5).iterrows():
            lines.append(f"- {row['term']}: {int(row['count'])}")

    if not top_diagnoses.empty:
        lines.append("")
        lines.append("Top diagnoses (all reports):")
        for _, row in top_diagnoses.head(5).iterrows():
            lines.append(f"- {row['diagnosis']}: {int(row['count'])}")

    if not top_delir_diagnoses.empty:
        lines.append("")
        lines.append("Top delir-related diagnosis keywords:")
        for _, row in top_delir_diagnoses.head(5).iterrows():
            lines.append(f"- {row['keyword']}: {int(row['count'])}")

    if not signal_freq.empty:
        lines.append("")
        lines.append("Most frequent extracted signal categories:")
        for _, row in signal_freq.head(5).iterrows():
            lines.append(f"- {row['signal_category']}: {int(row['count'])}")

    lines.append("")
    lines.append("Artifacts")
    lines.append(f"- tables: {EXPLORATION_TABLES_DIR}")
    lines.append(f"- plots: {EXPLORATION_PLOTS_DIR}")
    lines.append("")
    lines.append("Processing note: bertyp == Dokumentationsblatt excluded from exploration counts/plots.")
    EXPLORATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_validation_exploration_charts(
    raw_berichte: pd.DataFrame,
    baseline: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Report-type and baseline distributions for validation-oriented exploration."""
    if not raw_berichte.empty and "bertyp" in raw_berichte.columns:
        vc = raw_berichte["bertyp"].map(normalize_bertyp).value_counts()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(vc.index.astype(str), vc.values.astype(int), color="#2563eb")
        ax.set_title("Report type distribution (Dokumentationsblatt excluded)")
        ax.set_ylabel("Row count")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "report_type_distribution.png", dpi=120)
        plt.close(fig)

    if not predictions.empty and "bertyp" in predictions.columns and "klasse" in predictions.columns:
        pred = predictions.copy()
        pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
        pred["klasse"] = pd.to_numeric(pred["klasse"], errors="coerce").fillna(0).astype(int)
        rates = pred.groupby("bertyp")["klasse"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(rates.index.astype(str), (rates * 100).values, color="#16a34a")
        ax.set_title("Positive klasse rate by report type (report-level)")
        ax.set_ylabel("Percent klasse=1")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "report_type_positive_rates.png", dpi=120)
        plt.close(fig)

    if not baseline.empty and "baseline_composite" in baseline.columns:
        vc = pd.to_numeric(baseline["baseline_composite"], errors="coerce").fillna(0).astype(int).value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["0", "1"], [int(vc.get(0, 0)), int(vc.get(1, 0))], color=["#64748b", "#dc2626"])
        ax.set_title("Composite baseline distribution (patient-level)")
        ax.set_ylabel("Patients")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "composite_baseline_distribution.png", dpi=120)
        plt.close(fig)

    if not predictions.empty and "manual_review_candidate" in predictions.columns:
        s = predictions["manual_review_candidate"].astype(str).str.strip().str.lower()
        n_yes = int(s.isin(("1", "true", "yes")).sum())
        n_no = len(s) - n_yes
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["no", "yes"], [n_no, n_yes], color=["#94a3b8", "#f59e0b"])
        ax.set_title("manual_review_candidate (report-level)")
        ax.set_ylabel("Reports")
        fig.tight_layout()
        fig.savefig(EXPLORATION_PLOTS_DIR / "manual_review_candidate_distribution.png", dpi=120)
        plt.close(fig)

    if PATIENT_REPORTTYPE_MATRIX_PATH.exists():
        matrix = pd.read_csv(PATIENT_REPORTTYPE_MATRIX_PATH)
        matrix.to_csv(EXPLORATION_TABLES_DIR / "patient_reporttype_matrix_summary.csv", index=False)


def main() -> None:
    EXPLORATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EXPLORATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLORATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _warn_legacy_diagnosis_missing()

    reports = _load_reports_for_exploration()
    raw_berichte = _load_raw_berichte_optional()
    legacy_diag = _load_legacy_diagnosis_optional()
    icd10 = _safe_load(ICD10_PATH)
    icdsc = _safe_load(ICDSC_PATH)
    baseline = _load_structured_baseline_for_exploration()
    predictions = _load_predictions()

    if BERICHTE_INPUT_PATH.exists() and STRUCTURED_BASELINE_PATH.exists():
        try:
            _, _, cohort_counts, b_path, m_path = load_and_compute_current_cohort_counts()
            print_current_cohort_counts(cohort_counts, berichte_path=b_path, baseline_path=m_path)
        except Exception as exc:
            LOGGER.warning("Could not compute Berichte/baseline cohort counts: %s", exc)

    _write_overview_tables(reports, icd10, icdsc, baseline)
    _berichte_sections_exploration(raw_berichte)
    if not legacy_diag.empty:
        _diagnosis_exploration(legacy_diag)
    _icd10_exploration(icd10)
    _icdsc_exploration(icdsc)
    activity_berichte = raw_berichte if not raw_berichte.empty else reports
    _patient_activity_tables(activity_berichte, icd10, icdsc)
    _plot_report_length_distribution(reports)
    top_terms = _plot_keyword_frequency_from_reports(reports)
    top_diagnoses = _plot_top_diagnoses(reports, EXPLORATION_TABLES_DIR)
    top_delir_diagnoses = _plot_delir_diagnoses(reports, EXPLORATION_TABLES_DIR)
    signal_freq = _plot_signal_category_frequencies(predictions, EXPLORATION_TABLES_DIR)
    _plot_class_distribution_if_available(predictions)
    _plot_validation_exploration_charts(raw_berichte, baseline, predictions)
    _write_optional_token_frequency(reports)
    _write_exploration_summary(
        icd10,
        icdsc,
        baseline,
        reports,
        predictions,
        top_terms,
        top_diagnoses,
        top_delir_diagnoses,
        signal_freq,
        legacy_diag,
    )

    print(f"Exploration tables: {EXPLORATION_TABLES_DIR}")
    print(f"Exploration plots:  {EXPLORATION_PLOTS_DIR}")
    print(f"Exploration report: {EXPLORATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
