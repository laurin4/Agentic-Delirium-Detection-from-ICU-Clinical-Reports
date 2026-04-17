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
MAX_REPORTS = None  # None = alle Berichte; int = nur erste N Berichte

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
PREPARED_DATA_DIR = OUTPUTS_DIR / "prepared"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Input paths per mode (single source of truth; no duplicated path logic)
_MODE_INPUTS = {
    "real": {
        "icd10": REAL_RAW_DIR / "ICD.csv",
        "icdsc": REAL_RAW_DIR / "ICDSC.csv",
        "diagnosis": REAL_RAW_DIR / "Diagnosenliste.csv",
    },
    "synthetic": {
        "icd10": STRUCTURED_RAW_DIR / "synthetic_icd10.csv",
        "icdsc": STRUCTURED_RAW_DIR / "synthetic_icdsc.csv",
        "diagnosis": DIAGNOSIS_EXAMPLES_DIR / "synthetic_diagnoses.csv",
    },
}

if DATA_MODE not in _MODE_INPUTS:
    raise ValueError(
        f"Invalid DATA_MODE='{DATA_MODE}'. Allowed values: {sorted(_MODE_INPUTS)}"
    )

_paths = _MODE_INPUTS[DATA_MODE]
ICD10_PATH = _paths["icd10"]
ICDSC_PATH = _paths["icdsc"]
DIAGNOSIS_INPUT_PATH = _paths["diagnosis"]
REPORT_ID_MAPPING_PATH = STRUCTURED_DIR / "report_patient_ids.csv"

STRUCTURED_BASELINE_PATH = BASELINE_DIR / "structured_baseline.csv"
REPORT_VS_BASELINE_PATH = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.csv"
EVALUATION_MULTICLASS_SUMMARY_PATH = EVALUATION_DIR / "evaluation_multiclass_summary.csv"
EVALUATION_CONFUSION_3CLASS_PATH = EVALUATION_DIR / "confusion_matrix_3class.csv"
PATIENT_LEVEL_REPORTS_PATH = PREPARED_DATA_DIR / "patient_level_reports.csv"
VALIDATION_RESULTS_CSV_PATH = VALIDATION_DIR / "validation_results.csv"
VALIDATION_SUMMARY_TXT_PATH = VALIDATION_DIR / "validation_summary.txt"
