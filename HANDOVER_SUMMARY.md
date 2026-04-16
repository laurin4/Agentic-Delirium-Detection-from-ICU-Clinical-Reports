# Handover Summary

## Project Purpose
- Detect ICU delirium from clinical diagnosis text using a 3-agent pipeline.
- Compare model predictions against an operational structured baseline (ICD10 + ICDSC).
- Provide reproducible validation, evaluation, and exploratory analysis outputs.

## Core Architecture
- **Agent 1**: extraction (`src/agents/extraction.py`)
- **Agent 2**: interpretation (rule/prompt; default prompt) (`src/agents/interpretation.py`, `src/agents/interpretation_llm.py`)
- **Agent 3**: classification into classes 0/1/2 (`src/agents/classification.py`)

Pipeline stages:
1. Prepare structured baseline (`src/pipeline/prepare_structured_data.py`)
2. Run text pipeline (`src/pipeline/run_pipeline.py`)
3. Compare predictions vs baseline (`src/pipeline/compare_reports_vs_baseline.py`)
4. Evaluate metrics/plots (`src/pipeline/evaluate_predictions.py`)
5. Validate input consistency (`src/validation/validate_inputs.py`)
6. Advanced exploration (`src/analysis/run_exploration.py`)
7. In-depth analysis (`src/analysis/run_analysis.py`)

## Data Sources (Current Default)
Centralized in `src/pipeline/paths.py`.

- `DATA_MODE = "real"` (default)
- Real input files:
  - `Data/Raw/Diagnosenliste.csv`
  - `Data/Raw/ICD.csv`
  - `Data/Raw/ICDSC.csv`

Optional synthetic mode:
- set `DATA_MODE = "synthetic"` in `paths.py`
- uses:
  - `data/structured/raw/synthetic_icd10.csv`
  - `data/structured/raw/synthetic_icdsc.csv`
  - `data/anonymized/beispiele/synthetic_diagnoses.csv`

## Important Logic
- Diagnosis preprocessing: builds one patient-level report per patient from diagnosis rows (`src/preprocessing/diagnosis_mapper.py`).
- Baseline classes (operational reference):
  - Class 2: valid F05* (excluding F05.1) **and** ICDSC delir flag = 1
  - Class 1: not class 2, but at least one of valid ICD10 / delir flag / max ICDSC >= 4
  - Class 0: none of the above

## Single Source of Path Truth
- `src/pipeline/paths.py` is the central config.
- Do not hardcode paths elsewhere.
- New analysis/exploration output dirs are also defined there.

## Most Important Commands

### Full one-command run
```bash
./scripts/run_all.sh
```

### Preflight (recommended before sensitive runs)
```bash
./scripts/preflight_check.sh
```

### Manual step-by-step
```bash
python -m src.pipeline.prepare_structured_data
python -m src.pipeline.run_pipeline
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions
python -m src.validation.validate_inputs
python -m src.analysis.run_exploration
python -m src.analysis.run_analysis
```

## Output Structure
- `outputs/baseline/` → structured baseline tables
- `outputs/predictions/` → Agent 1/2/3 outputs
- `outputs/comparisons/` → merged prediction vs baseline
- `outputs/evaluation/` → multiclass metrics, confusion matrices, error exports, plots
- `outputs/validation/` → validation checks (`validation_results.csv`, `validation_summary.txt`)
- `outputs/analysis/`
  - `exploration/` → raw-data EDA tables/plots/report
  - `tables/`, `plots/`, `reports/` → in-depth analytical views

## Docker / Ubuntu Notes
- Dockerfile exists at `docker/Dockerfile`.
- Build:
```bash
docker build -f docker/Dockerfile -t delirium-pipeline .
```
- Example run:
```bash
docker run --rm -it \
  -v "$(pwd)/Data/Raw:/app/Data/Raw" \
  -v "$(pwd)/outputs:/app/outputs" \
  delirium-pipeline \
  python -m src.pipeline.prepare_structured_data
```

## Sensitive Data Safety
- `data/` and `Data/` are gitignored.
- Keep real ICU data only local/server-side.
- Avoid committing raw input files.

## Current Known Practical State
- Code and tests are passing in the current local setup.
- Placeholder `Data/Raw` files exist for dry-run wiring; replace with real files on Ubuntu.
- If ICD files are empty, pipeline runs but baseline/evaluation quality is limited (expected).

## Quick Troubleshooting
- No reports processed (`Anzahl Berichte: 0`):
  - Check `Data/Raw/Diagnosenliste.csv` exists and has rows.
- Baseline empty:
  - Check `Data/Raw/ICD.csv` and `Data/Raw/ICDSC.csv` have rows and expected columns.
- Format issues:
  - Reader supports `.csv`, `.xlsx`, `.xls` via `src/pipeline/tabular_io.py`.
