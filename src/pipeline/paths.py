import os
from pathlib import Path

# Projektbasis
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Daten
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_DIR
REAL_RAW_DIR = REAL_DATA_DIR / "raw"
ANONYMIZED_DIR = DATA_DIR / "anonymized"
DIAGNOSIS_EXAMPLES_DIR = ANONYMIZED_DIR / "beispiele"
STRUCTURED_DIR = DATA_DIR / "structured"
STRUCTURED_RAW_DIR = STRUCTURED_DIR / "raw"

# Default production inputs (CSV unter data/raw fuer Ubuntu/local parity).
# Set DATA_MODE = "synthetic" only for offline regression tests (CSV generator outputs).
DATA_MODE = "real"  # allowed: "real", "synthetic"


# Default cap (matches historical paths.py): avoids accidentally running the full corpus.
DEFAULT_MAX_REPORTS = 60


def parse_max_reports_env(raw: str | None = None) -> int | None:
    """
    Parse MAX_REPORTS from *raw* or from the environment.

    - Unset / blank → ``DEFAULT_MAX_REPORTS`` (30).
    - ``all`` (case-insensitive) → ``None`` (no limit, every patient-level report).
    - Otherwise a positive integer cap.

    Raises ValueError for invalid strings or non-positive integers (except ``all``).
    """
    if raw is None:
        raw = os.environ.get("MAX_REPORTS", "")
    raw = raw.strip()
    if not raw:
        return DEFAULT_MAX_REPORTS
    if raw.lower() == "all":
        return None
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (defaults to 30)."
        ) from exc
    if n <= 0:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (defaults to 30)."
        )
    return n


def _max_reports_from_environment() -> int | None:
    """``MAX_REPORTS`` env (see ``parse_max_reports_env``); defaults to first 30 reports."""
    return parse_max_reports_env()


MAX_REPORTS = _max_reports_from_environment()

# TEMPORARY PRESENTATION MODE — baseline_composite definition (revert for thesis default):
#   "OR"  = (ICDSC>=4) OR ICD10  — broader / sensitive thesis baseline
#   "AND" = (ICDSC>=4) AND ICD10 — stricter "secure delir cases" for presentation/demo
BASELINE_COMPOSITE_MODE = "AND"  # allowed: "OR", "AND"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
VALIDATION_DIR = OUTPUTS_DIR / "validation"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
ANALYSIS_TABLES_DIR = ANALYSIS_DIR / "tables"
ANALYSIS_PLOTS_DIR = ANALYSIS_DIR / "plots"
ANALYSIS_REPORTS_DIR = ANALYSIS_DIR / "reports"
EXPLORATION_DIR = ANALYSIS_DIR / "exploration"
EXPLORATION_TABLES_DIR = EXPLORATION_DIR / "tables"
EXPLORATION_PLOTS_DIR = EXPLORATION_DIR / "plots"
EXPLORATION_REPORTS_DIR = EXPLORATION_DIR / "reports"
ANALYSIS_EVALUATION_DIR = ANALYSIS_DIR / "evaluation"
ANALYSIS_EVALUATION_TABLES_DIR = ANALYSIS_EVALUATION_DIR / "tables"
ANALYSIS_EVALUATION_PLOTS_DIR = ANALYSIS_EVALUATION_DIR / "plots"
PREPARED_DATA_DIR = OUTPUTS_DIR / "prepared"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Final production raw inputs (DATA_MODE=real): Berichte.csv, ICD.csv, ICDSC.csv only.
# Legacy Diagnosenliste.csv is not used in the active pipeline (see LEGACY_DIAGNOSIS_INPUT_PATH).

# Input paths per mode (single source of truth; no duplicated path logic)
_MODE_INPUTS = {
    "real": {
        "icd10": REAL_RAW_DIR / "ICD.csv",
        "icdsc": REAL_RAW_DIR / "ICDSC.csv",
        "berichte_csv": REAL_RAW_DIR / "Berichte.csv",
    },
    "synthetic": {
        "icd10": STRUCTURED_RAW_DIR / "synthetic_icd10.csv",
        "icdsc": STRUCTURED_RAW_DIR / "synthetic_icdsc.csv",
        "berichte_csv": STRUCTURED_RAW_DIR / "synthetic_berichte.csv",
        # Legacy text source for offline regression only (INPUT_MODE=diagnosis).
        "diagnosis": DIAGNOSIS_EXAMPLES_DIR / "synthetic_diagnoses.csv",
    },
}

# Legacy path (not required for production). Former Diagnosenliste.csv location.
LEGACY_DIAGNOSIS_INPUT_PATH = REAL_RAW_DIR / "Diagnosenliste.csv"

if DATA_MODE not in _MODE_INPUTS:
    raise ValueError(
        f"Invalid DATA_MODE='{DATA_MODE}'. Allowed values: {sorted(_MODE_INPUTS)}"
    )

_paths = _MODE_INPUTS[DATA_MODE]
ICD10_PATH = _paths["icd10"]
ICDSC_PATH = _paths["icdsc"]
BERICHTE_INPUT_PATH = _paths["berichte_csv"]
# Legacy diagnosis list — only defined in synthetic mode; use Berichte.csv in production.
DIAGNOSIS_INPUT_PATH = _paths.get("diagnosis")
REPORT_ID_MAPPING_PATH = STRUCTURED_DIR / "report_patient_ids.csv"

STRUCTURED_BASELINE_PATH = BASELINE_DIR / "structured_baseline.csv"
REPORT_VS_BASELINE_PATH = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
REPORT_VS_BASELINE_EXCLUDED_PATH = (
    COMPARISONS_DIR / "report_vs_baseline_excluded_missing_baseline.csv"
)
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.csv"
EVALUATION_MULTICLASS_SUMMARY_PATH = EVALUATION_DIR / "evaluation_multiclass_summary.csv"
EVALUATION_CONFUSION_3CLASS_PATH = EVALUATION_DIR / "confusion_matrix_3class.csv"
EVALUATION_BINARY_BASELINES_DIR = EVALUATION_DIR / "binary_baselines"
EVALUATION_BINARY_BASELINES_TABLES_DIR = EVALUATION_BINARY_BASELINES_DIR / "tables"
EVALUATION_BINARY_BASELINES_PLOTS_DIR = EVALUATION_BINARY_BASELINES_DIR / "plots"
EVALUATION_BINARY_BASELINE_SUMMARY_PATH = (
    EVALUATION_BINARY_BASELINES_TABLES_DIR / "binary_baseline_summary.csv"
)
EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH = (
    EVALUATION_BINARY_BASELINES_TABLES_DIR / "binary_baseline_confusion_counts.csv"
)
EVALUATION_BINARY_BASELINE_REPORT_PATH = (
    EVALUATION_BINARY_BASELINES_DIR / "report.txt"
)
PATIENT_LEVEL_REPORTS_PATH = PREPARED_DATA_DIR / "patient_level_reports.csv"
VALIDATION_RESULTS_CSV_PATH = VALIDATION_DIR / "validation_results.csv"
VALIDATION_SUMMARY_TXT_PATH = VALIDATION_DIR / "validation_summary.txt"


LLM_DEBUG_DIR = OUTPUTS_DIR / "logs" / "llm_debug"
SQLITE_PREDICTIONS_DB_PATH = LOGS_DIR / "prediction_run.sqlite"

# Field-level keyword analysis (Berichte.csv vs structured baselines)
FIELD_DELIRIUM_ANALYSIS_DIR = ANALYSIS_DIR / "field_delirium"
FIELD_DELIRIUM_TABLES_DIR = FIELD_DELIRIUM_ANALYSIS_DIR / "tables"
FIELD_DELIRIUM_PLOTS_DIR = FIELD_DELIRIUM_ANALYSIS_DIR / "plots"

# Pre-model data coverage (Berichte vs structured baseline)
DATA_COVERAGE_ANALYSIS_DIR = ANALYSIS_DIR / "data_coverage"
DATA_COVERAGE_TABLES_DIR = DATA_COVERAGE_ANALYSIS_DIR / "tables"
DATA_COVERAGE_PLOTS_DIR = DATA_COVERAGE_ANALYSIS_DIR / "plots"

# Error review (legacy dir; manual review export uses MANUAL_REVIEW_DIR)
ERROR_REVIEW_DIR = ANALYSIS_DIR / "error_review"
ERROR_REVIEW_TABLES_DIR = ERROR_REVIEW_DIR / "tables"
ERROR_REVIEW_PLOTS_DIR = ERROR_REVIEW_DIR / "plots"

# Manual scientific review (TP/TN/FP/FN samples per primary baseline)
MANUAL_REVIEW_DIR = ANALYSIS_DIR / "manual_review"

# Patient-level validation aggregation (report-level predictions → patient matrix)
PATIENT_LEVEL_ANALYSIS_DIR = ANALYSIS_DIR / "patient_level"
PATIENT_REPORTTYPE_MATRIX_PATH = PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix.csv"
PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG = (
    PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix_preview.png"
)
PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF = (
    PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix_preview.pdf"
)

# Mixed manual validation sample (~100 patients)
MANUAL_VALIDATION_DIR = ANALYSIS_DIR / "manual_validation"
MANUAL_VALIDATION_SAMPLE_PATH = MANUAL_VALIDATION_DIR / "manual_validation_sample.csv"
MANUAL_ANNOTATION_SHEET_PATH = MANUAL_VALIDATION_DIR / "manual_annotation_sheet.csv"
MANUAL_ANNOTATION_SHEET_REPORT_PATH = (
    MANUAL_VALIDATION_DIR / "manual_annotation_sheet_report.txt"
)
PATIENT_VALIDATION_COHORT_PATH = MANUAL_VALIDATION_DIR / "patient_validation_cohort.csv"
PATIENT_VALIDATION_COHORT_REPORT_PATH = (
    MANUAL_VALIDATION_DIR / "patient_validation_cohort_report.txt"
)
MANUAL_VALIDATION_EVAL_DIR = MANUAL_VALIDATION_DIR / "evaluation"
MANUAL_REPORT_LABELS_PATH = MANUAL_VALIDATION_DIR / "manual_report_labels.csv"
FROZEN_VALIDATION_COHORT_DIR = MANUAL_VALIDATION_DIR / "frozen_validation_cohort"
FROZEN_PATIENT_VALIDATION_COHORT_PATH = (
    FROZEN_VALIDATION_COHORT_DIR / "patient_validation_cohort_frozen.csv"
)
FROZEN_MANUAL_REPORT_LABELS_PATH = (
    FROZEN_VALIDATION_COHORT_DIR / "manual_report_labels_frozen.csv"
)
FROZEN_COHORT_METADATA_PATH = FROZEN_VALIDATION_COHORT_DIR / "frozen_cohort_metadata.json"

# Presentation slide examples (report flow: excerpt → keywords → evidence → LLM → prediction)
PRESENTATION_EXAMPLES_DIR = ANALYSIS_DIR / "presentation_examples"
PRESENTATION_EXAMPLES_CSV_PATH = PRESENTATION_EXAMPLES_DIR / "presentation_examples.csv"
PRESENTATION_EXAMPLES_MD_PATH = PRESENTATION_EXAMPLES_DIR / "presentation_examples.md"
PRESENTATION_EXAMPLES_REPORT_PATH = (
    PRESENTATION_EXAMPLES_DIR / "presentation_examples_report.txt"
)

# Keyword / term association with predictions and baselines
KEYWORD_ANALYSIS_DIR = ANALYSIS_DIR / "keyword_analysis"
KEYWORD_ANALYSIS_TABLES_DIR = KEYWORD_ANALYSIS_DIR / "tables"
KEYWORD_ANALYSIS_PLOTS_DIR = KEYWORD_ANALYSIS_DIR / "plots"

# Field-level signal analysis (Berichte fields vs model / baselines)
FIELD_SIGNAL_ANALYSIS_DIR = ANALYSIS_DIR / "field_signal_analysis"
FIELD_SIGNAL_TABLES_DIR = FIELD_SIGNAL_ANALYSIS_DIR / "tables"
FIELD_SIGNAL_PLOTS_DIR = FIELD_SIGNAL_ANALYSIS_DIR / "plots"

# Evidence snippets (interpretability export; does not change predictions)
EVIDENCE_SNIPPETS_DIR = ANALYSIS_DIR / "evidence"
EVIDENCE_SNIPPETS_TABLES_DIR = EVIDENCE_SNIPPETS_DIR / "tables"
