from pathlib import Path

# Projektbasis
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Daten
DATA_DIR = PROJECT_ROOT / "data"
STRUCTURED_DIR = DATA_DIR / "structured"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Standard-Dateien
ICD10_PATH = STRUCTURED_DIR / "icd10.csv"
ICDSC_PATH = STRUCTURED_DIR / "icdsc.csv"
REPORT_ID_MAPPING_PATH = STRUCTURED_DIR / "report_patient_ids.csv"

STRUCTURED_BASELINE_PATH = BASELINE_DIR / "structured_baseline.csv"
REPORT_VS_BASELINE_PATH = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.csv"