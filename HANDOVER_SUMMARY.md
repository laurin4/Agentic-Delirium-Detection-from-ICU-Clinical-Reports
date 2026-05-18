# Handover Summary

## Project Purpose
- Detect ICU delirium from clinical diagnosis text using a 3-agent pipeline.
- Compare model predictions against an operational structured baseline (ICD10 + ICDSC).
- Provide reproducible validation, evaluation, and exploratory analysis outputs.

## Core Architecture
- **Pre-LLM layer**: structured rule-based evidence extraction (`src/preprocessing/evidence_extraction.py`) — full `report_text` is scanned; only bounded, section-tagged snippets are assembled for the LLM.
- **Agent 1**: extraction (`src/agents/extraction.py`) — JSON signal buckets from the **evidence bundle** (not the full report).
- **Agent 2**: interpretation (rule/prompt; default prompt) (`src/agents/interpretation.py`, `src/agents/interpretation_llm.py`) — assigns **signalstaerke** (`niedrig` / `mittel` / `hoch`).
- **Agent 3**: classification (`src/agents/classification.py`) — **binary** `klasse` 0/1 from signal strength (`mittel`/`hoch` → 1, `niedrig` → 0).
- **Prediction unit**: one row per **report** in `Berichte.csv` (report-level). Patient-level validation uses `patient_reporttype_matrix.csv`.
- **Excluded from processing** (raw CSV unchanged): `bertyp == Dokumentationsblatt` — logged as `excluded_dokumentationsblatt_count`.

## Evidence extraction (scientific / scalability)
- **Binary output only**: `klasse` ∈ {0, 1}. There is **no** multiclass prediction head.
- **Signal strength** remains `niedrig` | `mittel` | `hoch` (interpretation only); mapping to `klasse` is unchanged.
- The **entire** stitched `report_text` is scanned with deterministic keyword groups:
  - **direct_delir**, **indirect_symptom**, **negation**, **prophylaxis_or_risk** (see `evidence_extraction.py`).
- **Negated** delirium phrases are **not** treated as positive evidence; **prophylaxis / screening / risk-only** mentions are **not** auto-positive for delirium (the LLM is instructed; final class still flows through signal strength).
- If **no** snippet qualifies for LLM review (i.e. nothing beyond negation-only), the LLM is **skipped**, `llm_text_reduction_method=no_evidence_prefilter_skip`, and **`klasse=0`**.
- If actionable snippets exist, `llm_text_reduction_method=structured_evidence_extraction` and the LLM receives **`llm_report_text`**: labeled snippets + short instruction — **not** the full chart.
- **Transparency**: describe this two-stage design (rules → LLM) in thesis/defense materials; CSV stores structured `evidence_snippets` (JSON list) plus boolean flags for audit.
- **Clinical guardrails** (`src/agents/clinical_guardrails.py`, after Agent 2): hard-excludes only **no evidence**, **prophylaxis/risk-only**, and **negated delirium**. LLM positives from indirect symptoms or with alternative explanations are **kept as klasse=1** and flagged `manual_review_candidate=true` (not auto-downgraded).

### Environment (evidence + logging)
| Variable | Default | Role |
|----------|---------|------|
| `EVIDENCE_MAX_SNIPPETS` | 12 | Max structured snippets per patient. |
| `EVIDENCE_MAX_LLM_CHARS` | 8000 | Cap on assembled LLM evidence bundle size. |
| `EVIDENCE_WINDOW_SENTENCES` | 1 | Sentences before/after the hit sentence in each window. |
| `EVIDENCE_MAX_SNIPPET_CHARS` | 400 | Max characters per snippet `text` field. |
| `DEBUG_LLM_OUTPUT` | false | If true, print verbose per-agent debug (full previews, raw LLM). |

## Pipeline stages
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
- **Final production raw inputs** (semicolon-separated under `data/raw/`):
  - `Berichte.csv` — primary text (`PatientID`, clinical fields → `report_text`)
  - `ICD.csv` — `PatientID; icd_hd; icd_code`
  - `ICDSC.csv` — `PatientID; ICDSC_Max` (patient-level maximum score)
- **No** `Diagnosenliste.csv` in the active pipeline (`LEGACY_DIAGNOSIS_INPUT_PATH` only for documentation).

Optional synthetic mode (`DATA_MODE = "synthetic"`):
  - `data/structured/raw/synthetic_icd10.csv`, `synthetic_icdsc.csv`, `synthetic_berichte.csv`
  - Legacy: `data/anonymized/beispiele/synthetic_diagnoses.csv` (INPUT_MODE=diagnosis only)

## Important Logic
- **Baseline construction** (`prepare_structured_data`): ICD + ICDSC only → `structured_baseline.csv`.
- **ICD-10 delirium** (`has_delir_icd10` / `baseline_icd10`): main diagnosis `icd_hd == 1` and code in `F05.0`, `F05.8`, `F05.9` (excludes `F05.1`).
- **ICDSC** (`max_icdsc`): from `ICDSC_Max`; thresholds `baseline_icdsc_ge_*`, `baseline_icdsc_0`, `baseline_icdsc_1_to_3`, `baseline_icdsc_ge_4_grouped`.
- **Primary validation baseline** `baseline_composite` = `(baseline_icdsc_ge_4 == 1) OR (baseline_icd10 == 1)`.
- **Legacy** multiclass `baseline_reference_class` may still be written; primary evaluation uses binary baselines including `baseline_composite`.
- **Deprecated:** `Diagnosenliste.csv` / `diagnosis_mapper` — not used in production. `Berichte.csv` columns `diag`, `epikrise`, `jetziges_leiden`, `prozedere` map to report sections `[Diagnosen]`, `[Epikrise]`, etc.
- **Exploration** (`run_exploration.py`): Berichte + ICD + ICDSC + structured baseline + predictions; no crash when legacy diagnosis path is absent.

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
### Pre-prompt
export MAX_REPORTS=30
export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true

### Manual step-by-step
```bash
python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.pipeline.compare_reports_vs_baseline
python3 -m src.pipeline.evaluate_predictions
python3 -m src.validation.validate_inputs
python3 -m src.analysis.run_exploration
python3 -m src.analysis.run_analysis
python3 -m src.analysis.run_validation_suite
python3 -m src.analysis.create_patient_reporttype_matrix
python3 -m src.analysis.export_manual_validation_sample
```

## Validation outputs (patient-level)
- `outputs/analysis/patient_level/patient_reporttype_matrix.csv` — per-patient report-type positives + `baseline_composite` discrepancies.
- `outputs/analysis/manual_validation/manual_validation_sample.csv` — ~100 mixed patients for manual review (binary `manual_ground_truth`).
- Exploratory: `delir_probability_estimate` (0–100) in predictions CSV; not used for final `klasse`.

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
  - Check `data/raw/Berichte.csv` exists and has rows with `PatientID`.
- Baseline empty or wrong joins:
  - Re-run `python -m src.pipeline.prepare_structured_data`.
  - Check `data/raw/ICD.csv` (`PatientID; icd_hd; icd_code`) and `data/raw/ICDSC.csv` (`PatientID; ICDSC_Max`).
- Format issues:
  - Reader supports `.csv`, `.xlsx`, `.xls` via `src/pipeline/tabular_io.py`.
