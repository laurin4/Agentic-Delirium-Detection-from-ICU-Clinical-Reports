# PROJECT STATUS

## Project
Agentic Delirium Detection from ICU Clinical Reports

## Current State
The prototype pipeline is implemented and working on synthetic/anonymized example data.

### Completed
- Agent 1 implemented: extraction of delirium-related signals from text
- Agent 2 implemented in two versions:
  - rule-based interpretation
  - prompt-based interpretation
- Agent 3 implemented: final classification into
  - 0 = no delirium
  - 1 = possible delirium
  - 2 = documented delirium
- Structured data preparation pipeline implemented
- Baseline comparison pipeline implemented
- Evaluation pipeline implemented
- Basic automated tests implemented and passing
- Centralized path management implemented
- Output folders structured into:
  - outputs/baseline
  - outputs/predictions
  - outputs/comparisons
  - outputs/evaluation

## New Data Situation
Real project data now uses a single shared raw folder:
1. diagnosis list under `Data/Raw/Diagnosenliste.csv`
2. ICD10 under `Data/Raw/ICD.csv`
3. ICDSC under `Data/Raw/ICDSC.csv`

## Confirmed Design Decisions
- The diagnosis source will become the model input
- The agent pipeline remains unchanged conceptually
- Baseline should be redesigned as a 3-class operational reference system
- ICD10 delirium should include all `F05*` codes except `F05.1`
- ICDSC should be aggregated at patient level
- `ICDSC_Value >= 4` should contribute to possible delirium
- Final execution target is Ubuntu server in Docker
- The code must be written so that sensitive files remain local and are not exposed to external AI systems

## Baseline Logic (Current Agreed Version)
- Class 2 (documented delirium):
  - valid ICD10 delirium code (`F05*`, excluding `F05.1`)
  - AND any positive ICDSC delirium flag
- Class 1 (possible delirium):
  - not class 2, but at least one of:
    - valid ICD10 delirium code
    - any positive ICDSC delirium flag
    - max ICDSC value >= 4
- Class 0 (no delirium):
  - none of the above

## Next Technical Milestone
Adapt the project from synthetic text reports to real diagnosis-based patient reports:
1. parse and preprocess diagnosis data
2. build one patient-level report per stay
3. adapt baseline generation from real ICD10 + ICDSC
4. keep the existing agent pipeline as the main analysis engine
5. update evaluation and visual outputs for the 3-class baseline

## Cautions
- The diagnosis data may not be cleanly tabular
- ICD10 and ICDSC are operational clinical proxies, not perfect truth
- Sedation and underdocumentation remain important confounders
- All implementation must remain compatible with local/server-only execution


## Synthetic Data Generation (New)

A synthetic data generation module is introduced to enable full end-to-end testing of the pipeline without access to sensitive real data.

### Purpose
- Validate diagnosis preprocessing robustness
- Validate 3-class baseline logic
- Test full pipeline integration (Agent 1/2/3 + baseline + evaluation)
- Enable stress testing with scalable patient counts

### Generated Data
The generator creates three aligned datasets:
1. diagnosis data (patient-level clinical entries)
2. ICD10 data (billing/diagnosis codes)
3. ICDSC data (time-series delirium scores)

All datasets share consistent PatientIDs.

### Design Principles
- Includes temporal progression (multiple entries per patient)
- Includes clinical shorthand and noisy ICU-style text
- Includes clearly defined baseline scenarios:
  - documented delirium (class 2)
  - possible delirium (class 1)
  - no delirium (class 0)
- Includes edge cases such as:
  - ICD10 F05.1 (excluded delirium)
  - high ICDSC values without flags
  - discordant ICD10 vs ICDSC cases

### Next Step
Use the synthetic data to validate:
- diagnosis → report mapping
- baseline correctness
- pipeline stability


## Real File Switch (default inputs)

The repository now defaults to **`Data/Raw/` CSV** inputs for ICU delirium work (`DATA_MODE = "real"` in `paths.py`):

- `Data/Raw/Diagnosenliste.csv`
- `Data/Raw/ICD.csv`
- `Data/Raw/ICDSC.csv`

After copying the project to Ubuntu, place these three files at the paths above, install dependencies from `requirements.txt`, and run the same `python -m src.pipeline.*` commands as in the README. No pipeline rewrites are required when switching back to synthetic: set `DATA_MODE = "synthetic"` only.

---

## Data inputs (default: real CSV in `Data/Raw`)

Production defaults are defined in `src/pipeline/paths.py` (`DATA_MODE = "real"`):

| Role | Path |
|------|------|
| Diagnosis | `Data/Raw/Diagnosenliste.csv` |
| ICD10 | `Data/Raw/ICD.csv` |
| ICDSC | `Data/Raw/ICDSC.csv` |

Tabular loading (CSV or Excel) is centralized in `src/pipeline/tabular_io.py` (`read_tabular`). ICD tables still support `PatientID` → `PatientenID` normalization in `prepare_structured_data`.

### Optional synthetic mode (regression / CI)
- Set `DATA_MODE = "synthetic"` in `paths.py` to use generator outputs (`synthetic_icd10.csv`, `synthetic_icdsc.csv`, `synthetic_diagnoses.csv`) without changing pipeline modules.
- Generate them with: `python scripts/generate_synthetic_data.py`



## Validation and Evaluation Milestone (Completed)

### What was delivered
- **Centralized paths** (`src/pipeline/paths.py`): `VALIDATION_DIR`, `PREPARED_DATA_DIR`, `PATIENT_LEVEL_REPORTS_PATH`, multiclass evaluation artifact paths; switching synthetic/real remains `DATA_MODE` only.
- **Validation layer**: `python -m src.validation.validate_inputs` writes `outputs/validation/validation_results.csv` and `validation_summary.txt` (row counts, duplicate/missing IDs, diagnosis vs ICD10/ICDSC patient-set consistency, optional structured baseline distribution).
- **Multiclass evaluation** (`src/pipeline/evaluate_predictions.py`): primary metrics use `klasse` vs `baseline_reference_class`; outputs include `confusion_matrix_3class.csv`, `evaluation_multiclass_summary.csv`, `multiclass_per_class_metrics.csv`, ordinal over/under-call counts, and error exports (`errors_exact_mismatch.csv`, `errors_overcall_pred_gt_reference.csv`, `errors_undercall_pred_lt_reference.csv`). **Binary** documented-delirium metrics and legacy crosstabs are **secondary** (`*_binary_secondary.csv`, `evaluation_binary_secondary_summary.csv`).
- **Plots** (matplotlib, reproducible): `outputs/evaluation/plots/` — class distribution, 3-class confusion heatmap, jittered prediction vs reference scatter. `MPLCONFIGDIR` defaults under `outputs/.mplconfig` for headless/Docker-friendly runs.
- **ICD column normalization**: `prepare_structured_data.load_data()` maps `PatientID` → `PatientenID` when needed so schemas align without downstream hacks.
- **Tabular I/O**: `read_tabular()` supports `.csv` and `.xlsx` / `.xls` for ICD and diagnosis inputs.

### How to run (after predictions + comparison exist)
```bash
python -m src.pipeline.prepare_structured_data
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions
python -m src.validation.validate_inputs
python -m src.analysis.run_exploration
python -m src.analysis.run_analysis
```

### Design requirement (unchanged)
Synthetic vs real: change only `DATA_MODE` (and place files under the paths defined in `paths.py`); pipeline entrypoints stay the same.

## In-depth Analysis Pipeline (New)

A dedicated analysis/exploration layer now exists separate from core evaluation:

- Entry point: `python -m src.analysis.run_analysis`
- Output root: `outputs/analysis/`
  - `tables/`:
    - `input_quality_checks.csv`
    - source snapshots (`diagnosis_raw_snapshot.csv`, `icd10_raw_snapshot.csv`, `icdsc_raw_snapshot.csv`)
    - class distributions
    - patient-level analysis table
    - confusion/error detail tables
  - `plots/`:
    - predicted vs reference class distribution
    - signal strength composition by predicted class
    - 3-class confusion heatmap
    - error vs extracted hits scatter
    - report length histogram
  - `reports/analysis_summary.txt`:
    - concise narrative summary for quick interpretation

## Advanced Data Exploration (New)

Dedicated raw-input EDA is now available:

- Entry point: `python -m src.analysis.run_exploration`
- Output root: `outputs/analysis/exploration/`
  - `tables/`: dataset overview, missingness, patient-set overlaps, top diagnosis entries/terms, ICD10 code frequencies, ICDSC summaries, temporal distributions, per-patient activity
  - `plots/`: top diagnosis terms, top ICD10 codes, ICDSC histogram, patient-activity boxplot
  - `reports/exploration_summary.txt`: compact human-readable EDA narrative