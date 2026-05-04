from pathlib import Path
import re
import pandas as pd

from src.pipeline.paths import OUTPUTS_DIR, DIAGNOSIS_INPUT_PATH, STRUCTURED_BASELINE_PATH
from src.preprocessing.diagnosis_mapper import build_patient_level_reports


OVERLAP_DIR = OUTPUTS_DIR / "analysis" / "overlap"
OVERLAP_TABLES_DIR = OVERLAP_DIR / "tables"
OVERLAP_REPORTS_DIR = OVERLAP_DIR / "reports"


DELIR_TEXT_PATTERNS = [
    r"\bdelir\b",
    r"\bdelirium\b",
    r"\bdelirant\b",
    r"\bdelirös\b",
    r"\bhypoaktives delir\b",
    r"\bhyperaktives delir\b",
]


def _has_text_delir(text: str) -> bool:
    text = str(text or "").lower()
    return any(re.search(pattern, text) for pattern in DELIR_TEXT_PATTERNS)


def _load_text_reports() -> pd.DataFrame:
    df = build_patient_level_reports(DIAGNOSIS_INPUT_PATH).copy()
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()
    df["has_text_delir"] = df["report_text"].apply(_has_text_delir)
    return df[["PatientenID", "bericht", "report_text", "has_text_delir"]]


def _load_baseline() -> pd.DataFrame:
    df = pd.read_csv(STRUCTURED_BASELINE_PATH).copy()
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    df["has_icd10_delir"] = df["has_main_delir_icd10"].fillna(0).astype(int) == 1
    df["has_icdsc_delir"] = pd.to_numeric(df["max_icdsc"], errors="coerce").fillna(0) >= 4

    return df[["PatientenID", "has_icd10_delir", "has_icdsc_delir", "max_icdsc"]]


def _build_overlap_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["all_three"] = df["has_icd10_delir"] & df["has_icdsc_delir"] & df["has_text_delir"]
    df["icd10_only"] = df["has_icd10_delir"] & ~df["has_icdsc_delir"] & ~df["has_text_delir"]
    df["icdsc_only"] = df["has_icdsc_delir"] & ~df["has_icd10_delir"] & ~df["has_text_delir"]
    df["text_only"] = df["has_text_delir"] & ~df["has_icd10_delir"] & ~df["has_icdsc_delir"]

    df["icd10_icdsc_only"] = df["has_icd10_delir"] & df["has_icdsc_delir"] & ~df["has_text_delir"]
    df["icd10_text_only"] = df["has_icd10_delir"] & df["has_text_delir"] & ~df["has_icdsc_delir"]
    df["icdsc_text_only"] = df["has_icdsc_delir"] & df["has_text_delir"] & ~df["has_icd10_delir"]

    df["none_positive"] = ~df["has_icd10_delir"] & ~df["has_icdsc_delir"] & ~df["has_text_delir"]

    return df


def _write_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("n_patients", len(df)),
        ("icd10_delir", int(df["has_icd10_delir"].sum())),
        ("icdsc_delir_ge4", int(df["has_icdsc_delir"].sum())),
        ("text_delir", int(df["has_text_delir"].sum())),
        ("icd10_and_icdsc", int((df["has_icd10_delir"] & df["has_icdsc_delir"]).sum())),
        ("icd10_and_text", int((df["has_icd10_delir"] & df["has_text_delir"]).sum())),
        ("icdsc_and_text", int((df["has_icdsc_delir"] & df["has_text_delir"]).sum())),
        ("all_three", int(df["all_three"].sum())),
        ("icd10_only", int(df["icd10_only"].sum())),
        ("icdsc_only", int(df["icdsc_only"].sum())),
        ("text_only", int(df["text_only"].sum())),
        ("icd10_icdsc_only", int(df["icd10_icdsc_only"].sum())),
        ("icd10_text_only", int(df["icd10_text_only"].sum())),
        ("icdsc_text_only", int(df["icdsc_text_only"].sum())),
        ("none_positive", int(df["none_positive"].sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "count"])


def main():
    OVERLAP_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline()
    text_reports = _load_text_reports()

    merged = baseline.merge(text_reports, on="PatientenID", how="outer")
    merged["has_icd10_delir"] = merged["has_icd10_delir"].fillna(False).astype(bool)
    merged["has_icdsc_delir"] = merged["has_icdsc_delir"].fillna(False).astype(bool)
    merged["has_text_delir"] = merged["has_text_delir"].fillna(False).astype(bool)

    overlap = _build_overlap_table(merged)
    summary = _write_summary(overlap)

    overlap.to_csv(OVERLAP_TABLES_DIR / "patient_level_overlap.csv", index=False)
    summary.to_csv(OVERLAP_TABLES_DIR / "overlap_summary.csv", index=False)

    with open(OVERLAP_REPORTS_DIR / "overlap_report.txt", "w", encoding="utf-8") as f:
        f.write("Overlap analysis across ICD10, ICDSC, and diagnosis text\n")
        f.write("=" * 60 + "\n\n")
        for _, row in summary.iterrows():
            f.write(f"{row['metric']}: {row['count']}\n")

    print(f"Overlap tables: {OVERLAP_TABLES_DIR}")
    print(f"Overlap report: {OVERLAP_REPORTS_DIR / 'overlap_report.txt'}")


if __name__ == "__main__":
    main()